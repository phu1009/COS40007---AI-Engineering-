"""
COS40007 Group 3 - Anti-Spoofing Login: training script for the GitHub Actions pipeline
=======================================================================================

This is the model training step of our project (face presentation attack detection,
"PAD", for a camera-based login that runs on a Raspberry Pi). GitHub Actions runs it
on every push to main and uploads its outputs as artefacts.

What it does (same stages as our system blueprint, kept small so CI finishes in minutes):
  1. Data        - builds a small synthetic face dataset with three classes of capture:
                   live (bona fide), print attack and screen-replay attack.
                   The real datasets (CelebA-Spoof, Replay-Attack) are licensed and tens
                   of GB, so they cannot be downloaded inside CI; this stand-in keeps
                   the pipeline runnable and will be swapped for the DVC-tracked data.
  2. Split       - subject-disjoint train / validation / test split (no person in two sets).
  3. Model       - lightweight CNN (depthwise-separable convolutions, MiniFASNet-style)
                   that outputs a live score in [0, 1].
  4. Threshold   - chosen on VALIDATION data so that BPCER <= 5 %.
  5. Evaluation  - ISO/IEC 30107-3 metrics on the held-out TEST set:
                   APCER (per attack type), BPCER, ACER, plus accuracy and ROC-AUC.
  6. Outputs     - model_results.png, metrics.txt and pad_model.keras.

Run locally with:  python model.py
"""

# Suppress TensorFlow info messages
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import time

import matplotlib

matplotlib.use("Agg")  # no screen on the GitHub runner
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ============================================================================
# SETTINGS
# ============================================================================
SEED = 42
IMG_SIZE = 64            # input size of the PAD model (pixels)
N_SUBJECTS = 100         # synthetic people
CAPTURES = 6             # captures per person and per class
SPLIT = (0.6, 0.2, 0.2)  # train / validation / test, split BY SUBJECT
EPOCHS = 15
BATCH_SIZE = 32
TARGET_BPCER = 0.05      # operating point chosen on validation data
CLASS_NAMES = ["live", "print", "replay"]

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)

# ============================================================================
# 1. SYNTHETIC FACE DATA (live / print attack / replay attack)
# ============================================================================
yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE] / IMG_SIZE


def make_subject():
    """Random appearance parameters for one synthetic person."""
    return {
        "skin": rng.uniform([0.55, 0.40, 0.30], [0.95, 0.75, 0.60]),
        "bg": rng.uniform(0.1, 0.9, size=3),
        "face_w": rng.uniform(0.28, 0.36),
        "face_h": rng.uniform(0.36, 0.44),
        "eye_gap": rng.uniform(0.12, 0.18),
    }


def render_face(s):
    """Draw a simple face (ellipse, eyes, mouth) with random pose and lighting."""
    cx, cy = 0.5 + rng.normal(0, 0.03), 0.5 + rng.normal(0, 0.03)
    face = ((xx - cx) / s["face_w"]) ** 2 + ((yy - cy) / s["face_h"]) ** 2 <= 1
    img = np.ones((IMG_SIZE, IMG_SIZE, 3)) * s["bg"]
    img[face] = s["skin"]
    for ex in (cx - s["eye_gap"], cx + s["eye_gap"]):  # eyes
        eye = ((xx - ex) / 0.05) ** 2 + ((yy - (cy - 0.08)) / 0.03) ** 2 <= 1
        img[eye] = 0.15
    mouth = ((xx - cx) / 0.12) ** 2 + ((yy - (cy + 0.18)) / 0.03) ** 2 <= 1
    img[mouth] = [0.55, 0.2, 0.2]
    light = 0.75 + 0.5 * rng.uniform() * (xx * rng.choice([-1, 1]) + 0.5)  # side light
    return img * light[..., None]


def make_live(s):
    """Bona fide capture: real face + fine camera sensor noise."""
    return render_face(s) + rng.normal(0, 0.03, (IMG_SIZE, IMG_SIZE, 3))


def make_print(s):
    """Print attack: halftone dot pattern, washed-out colours, slight paper tint."""
    img = render_face(s)
    period = rng.uniform(3.0, 5.0)
    strength = rng.uniform(0.04, 0.12)
    dots = np.sin(2 * np.pi * xx * IMG_SIZE / period) * np.sin(2 * np.pi * yy * IMG_SIZE / period)
    gray = img.mean(axis=2, keepdims=True)
    img = 0.6 * img + 0.4 * gray + strength * dots[..., None] + [0.04, 0.03, 0.0]
    return img + rng.normal(0, 0.02, img.shape)


def make_replay(s):
    """Screen-replay attack: moire stripes, bluish screen cast and a glare spot."""
    img = render_face(s)
    angle = rng.uniform(0, np.pi)
    freq = rng.uniform(6, 14)
    strength = rng.uniform(0.04, 0.12)
    moire = np.sin(2 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle)))
    gx, gy = rng.uniform(0.2, 0.8, size=2)
    glare = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / 0.01)) * rng.uniform(0.1, 0.4)
    img = img * [0.92, 0.97, 1.08] + strength * moire[..., None] + glare[..., None]
    return img + rng.normal(0, 0.02, img.shape)


def build_dataset():
    """Return images X, labels y (1 = live), attack type and subject id."""
    makers = [make_live, make_print, make_replay]
    X, y, kind, subj = [], [], [], []
    for sid in range(N_SUBJECTS):
        s = make_subject()
        for k, make in enumerate(makers):
            for _ in range(CAPTURES):
                X.append(make(s))
                y.append(1 if k == 0 else 0)
                kind.append(k)
                subj.append(sid)
    X = np.clip(np.array(X, dtype=np.float32), 0.0, 1.0)  # normalised to [0, 1]
    return X, np.array(y, dtype=np.float32), np.array(kind), np.array(subj)


# ============================================================================
# 2. SUBJECT-DISJOINT SPLIT
# ============================================================================
def subject_split(subj):
    ids = rng.permutation(N_SUBJECTS)
    n_tr, n_va = int(SPLIT[0] * N_SUBJECTS), int(SPLIT[1] * N_SUBJECTS)
    groups = ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]
    masks = [np.isin(subj, g) for g in groups]
    # leakage check: no subject may appear in two sets
    assert not (set(groups[0]) & set(groups[1]) or set(groups[0]) & set(groups[2]) or set(groups[1]) & set(groups[2]))
    return masks


# ============================================================================
# 3. MODEL - lightweight CNN suitable for a Raspberry Pi
# ============================================================================
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.SeparableConv2D(32, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.SeparableConv2D(64, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid", name="live_score"),
    ], name="pad_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ============================================================================
# 4-5. THRESHOLD AND PAD METRICS (ISO/IEC 30107-3)
# ============================================================================
def choose_threshold(scores, labels, target_bpcer=TARGET_BPCER):
    """Highest threshold whose BPCER on validation data is <= target."""
    live = np.sort(scores[labels == 1])
    k = int(np.floor(target_bpcer * len(live)))  # number of live samples we may reject
    return float(live[k]) if len(live) else 0.5


def pad_metrics(scores, labels, kind, thr):
    """APCER per attack type, BPCER, ACER and accuracy at threshold thr."""
    pred_live = scores >= thr
    bpcer = float(np.mean(~pred_live[labels == 1]))
    apcer = {CLASS_NAMES[k]: float(np.mean(pred_live[kind == k])) for k in (1, 2)}
    apcer_max = max(apcer.values())
    acc = float(np.mean(pred_live == (labels == 1)))
    return {"APCER": apcer, "APCER_max": apcer_max, "BPCER": bpcer,
            "ACER": (apcer_max + bpcer) / 2, "accuracy": acc}


def roc_curve(scores, labels):
    """False/true positive rates with 'live' as the positive class."""
    order = np.argsort(-scores)
    lab = labels[order]
    tpr = np.concatenate([[0], np.cumsum(lab) / max(lab.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(1 - lab) / max((1 - lab).sum(), 1)])
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


# ============================================================================
# 6. OUTPUTS
# ============================================================================
def plot_results(history, X, kind, scores, labels, thr, m, fpr, tpr, auc, run_label):
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))

    # (a) sample images
    samples = [np.where(kind == k)[0][:3] for k in range(3)]
    grid = np.concatenate([np.concatenate([X[i] for i in idx], axis=1) for idx in samples], axis=0)
    ax[0, 0].imshow(grid)
    ax[0, 0].set_yticks([IMG_SIZE * (k + 0.5) for k in range(3)], CLASS_NAMES)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_title("Test samples (rows: live / print / replay)", fontweight="bold")

    # (b) loss and (c) AUC per epoch
    for a, key, name in ((ax[0, 1], "loss", "Loss"), (ax[0, 2], "auc", "ROC-AUC")):
        a.plot(history.history[key], label="Training", linewidth=2)
        a.plot(history.history["val_" + key], label="Validation", linewidth=2)
        a.set_title(f"{name} per epoch", fontweight="bold")
        a.set_xlabel("Epoch")
        a.legend()
        a.grid(True, alpha=0.3)

    # (d) ROC curve with the chosen operating point
    ax[1, 0].plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    ax[1, 0].plot([0, 1], [0, 1], "--", color="grey")
    ax[1, 0].scatter([m["APCER_max"]], [1 - m["BPCER"]], color="red", zorder=3,
                     label=f"Threshold {thr:.2f}")
    ax[1, 0].set_xlabel("Attacks accepted as live (APCER)")
    ax[1, 0].set_ylabel("Live users accepted (1 - BPCER)")
    ax[1, 0].set_title("ROC curve - test set", fontweight="bold")
    ax[1, 0].legend(loc="lower right")
    ax[1, 0].grid(True, alpha=0.3)

    # (e) score distribution per class
    for k, colour in zip(range(3), ("tab:green", "tab:orange", "tab:red")):
        ax[1, 1].hist(scores[kind == k], bins=20, range=(0, 1), alpha=0.6, color=colour, label=CLASS_NAMES[k])
    ax[1, 1].axvline(thr, color="black", linestyle="--", label="Threshold")
    ax[1, 1].set_xlabel("Live score")
    ax[1, 1].set_title("Score distribution - test set", fontweight="bold")
    ax[1, 1].legend()

    # (f) confusion matrix at the threshold
    pred_live = scores >= thr
    cm = np.array([[np.sum(pred_live & (labels == 1)), np.sum(~pred_live & (labels == 1))],
                   [np.sum(pred_live & (labels == 0)), np.sum(~pred_live & (labels == 0))]])
    ax[1, 2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[1, 2].text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=16,
                          color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[1, 2].set_xticks([0, 1], ["Predicted live", "Predicted attack"])
    ax[1, 2].set_yticks([0, 1], ["Live", "Attack"])
    ax[1, 2].set_title(f"Confusion matrix (ACER = {m['ACER']:.2%})", fontweight="bold")

    fig.suptitle(f"COS40007 Group 3 - Anti-Spoofing PAD model  |  {run_label}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("model_results.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("Plot saved to model_results.png")


def save_metrics(m, auc, thr, sizes, seconds, run_label, filename="metrics.txt"):
    lines = [
        "=" * 60,
        "COS40007 GROUP 3 - ANTI-SPOOFING (PAD) MODEL METRICS",
        "=" * 60,
        f"Run                 : {run_label}",
        f"Commit              : {os.environ.get('GITHUB_SHA', 'local')[:7]}",
        f"TensorFlow version  : {tf.__version__}",
        f"Training time       : {seconds:.1f} s",
        "Data                : synthetic stand-in (live / print / replay), subject-disjoint split",
        f"Samples (train/val/test): {sizes[0]} / {sizes[1]} / {sizes[2]}",
        "-" * 60,
        f"Threshold (chosen on validation, BPCER <= {TARGET_BPCER:.0%}): {thr:.4f}",
        "Test-set results (ISO/IEC 30107-3):",
        f"  APCER print attack : {m['APCER']['print']:.2%}",
        f"  APCER replay attack: {m['APCER']['replay']:.2%}",
        f"  APCER (max)        : {m['APCER_max']:.2%}",
        f"  BPCER              : {m['BPCER']:.2%}",
        f"  ACER               : {m['ACER']:.2%}",
        f"  Accuracy           : {m['accuracy']:.2%}",
        f"  ROC-AUC            : {auc:.4f}",
        "=" * 60,
    ]
    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Metrics saved to {filename}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    run_label = (f"GitHub Actions run #{os.environ['GITHUB_RUN_NUMBER']}"
                 if "GITHUB_RUN_NUMBER" in os.environ else "local run")
    print(f"TensorFlow version: {tf.__version__} | {run_label}")

    # 1-2. data and subject-disjoint split
    X, y, kind, subj = build_dataset()
    tr, va, te = subject_split(subj)
    sizes = (int(tr.sum()), int(va.sum()), int(te.sum()))
    print(f"Dataset: {len(X)} images {X.shape[1:]} | train/val/test = {sizes}")

    # class weights: there are two attack captures for every live capture
    n_live, n_attack = y[tr].sum(), (1 - y[tr]).sum()
    class_weight = {0: len(y[tr]) / (2 * n_attack), 1: len(y[tr]) / (2 * n_live)}

    # 3. train
    model = build_model()
    model.summary()
    start = time.time()
    history = model.fit(
        X[tr], y[tr],
        validation_data=(X[va], y[va]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
        verbose=2,
    )
    seconds = time.time() - start

    # 4. threshold on validation data only
    val_scores = model.predict(X[va], verbose=0).ravel()
    thr = choose_threshold(val_scores, y[va])

    # 5. evaluate once on the test set
    test_scores = model.predict(X[te], verbose=0).ravel()
    m = pad_metrics(test_scores, y[te], kind[te], thr)
    fpr, tpr, auc = roc_curve(test_scores, y[te])

    # 6. outputs
    plot_results(history, X[te], kind[te], test_scores, y[te], thr, m, fpr, tpr, auc, run_label)
    save_metrics(m, auc, thr, sizes, seconds, run_label)
    model.save("pad_model.keras")
    print("Model saved to pad_model.keras")
    print("\nSCRIPT COMPLETED SUCCESSFULLY")
