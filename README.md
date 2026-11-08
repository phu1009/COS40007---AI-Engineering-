# COS40007 – Artificial Intelligence Engineering

[![Train ML Model](https://github.com/phu1009/COS40007---AI-Engineering-/actions/workflows/train.yml/badge.svg)](https://github.com/phu1009/COS40007---AI-Engineering-/actions/workflows/train.yml)

## Group Project – Group 3: Anti-Spoofing Login (Edge Liveness Detection on Raspberry Pi)

### Group Task 2 – GitHub Actions ML pipeline

```
git push ? GitHub Actions ? ubuntu-latest runner
        ? checkout ? set up Python 3.11 ? cache pip ? install requirements.txt
        ? python model.py  (trains the anti-spoofing model)
"        ? upload artefacts: model_results.png, metrics.txt, pad_model.keras"
```

| File | Purpose |
|---|---|
"| `.github/workflows/train.yml` | Pipeline definition. Triggers: push to `main`, pull request to `main`, manual run (`workflow_dispatch`). |"
"| `model.py` | Project training script: lightweight CNN that tells live faces from print and screen-replay attacks; reports APCER, BPCER, ACER and ROC-AUC. |"
"| `requirements.txt` | Python packages for the pipeline (TensorFlow, NumPy, Matplotlib). |"

"**Run it:** push to `main`, or open the **Actions** tab ? **Train ML Model** ? **Run workflow**."
**Get the results:** open the finished run ? **Artifacts** ? download **model-results**.

`model.py` uses a small synthetic stand-in dataset (live / print / replay) because the real datasets
"(CelebA-Spoof, Replay-Attack) are licensed and too large to download inside CI."

### Repo layout

| Folder | Content |
|---|---|
"| `Group Project/` | Group project documents (Task 1 report, system blueprint, Jira backlog) and code skeleton |"
| `Portfolio Assessment 1–3/` | Individual portfolio assessments |
