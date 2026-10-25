---
title: "Anti-Spoofing Login – Project Documentation"
subtitle: "COS40007 Artificial Intelligence Engineering · Group 3"
---

# 1. Project description

**Project:** Anti-Spoofing for Secure Login – Edge Liveness Detection on Raspberry Pi 5

We are building an anti-spoofing (presentation attack detection, PAD) component for a camera-based login flow. When a user logs in, a Raspberry Pi 5 with a Camera Module 3 checks that a live person is in front of the camera and not a printed photo, a photo shown on a phone or monitor, or a replayed video. The login only succeeds when the password check **and** the liveness check pass.

**Objectives**

- Build a reproducible pipeline: face detection → preprocessing → spoof detection → login decision.
- Compare baseline models and select one by attack acceptance (APCER), genuine-user rejection (BPCER) and latency on the Pi.
- Integrate the result into a web login prototype with clear user states: pass, spoof suspected, retry required, technical failure.
- Design the system so it can be monitored and retrained over time (MLOps).

**Scope (MoSCoW)**

| Priority | Items |
|---|---|
| Must | Passive liveness running on the Pi; web login demo; APCER/BPCER/ACER and latency report; fail-safe states |
| Should | Optional blink challenge as a second step; drift dashboard |
| Could | Camera Module 3 NoIR with infrared lighting; comparison with a Hailo AI HAT+ |
| Won't (this semester) | Face recognition / identity matching; native mobile app; 3D-mask attacks |

This is an academic prototype for a controlled demonstration. It does not guarantee identity and does not replace other authentication controls.

# 2. System overview

![System blueprint](architecture/system_blueprint.png)

The system has three layers on top of a shared DevOps/MLOps backbone:

1. **Offline ML pipeline (stages 1–6):** public datasets → ingestion and DVC versioning → preprocessing (YuNet face crop, resize, normalise, quality filter, subject-disjoint split) → feature engineering (CNN features, LBP baseline, augmentation) → training and selection → evaluation and packaging (ONNX INT8 + threshold config).
2. **Edge device – Raspberry Pi 5 (stages 7–11):** camera capture → on-device preprocessing (same code as training) → PAD inference with ONNX Runtime → decision engine (PASS / SPOOF SUSPECTED / RETRY / TECHNICAL FAILURE, fail-closed, signed result) → telemetry without images.
3. **Application and operations (stages 12–15):** web login UI → backend auth API (password check, nonce, signature verification, rate limits) → logging and monitoring → drift detection and retraining trigger.

**Login sequence:** ① user submits credentials → ② backend sends session + nonce to the Pi → ③ Pi returns a signed result → ④ backend returns the login decision → ⑤ attempt logged without images → ⑥ monitoring may trigger retraining.

**Datasets**

| Dataset | Use | Licence / access notes |
|---|---|---|
| CelebA-Spoof (625,537 images, 10,177 subjects, live:spoof ≈ 1:3) | Main training / validation / test | Non-commercial research only; no redistribution |
| Replay-Attack (Idiap; print, phone and tablet replay) | Cross-dataset test | EULA, must be requested by permanent academic staff (our tutor) |
| Pi-camera sanity set (team members, with consent) | Test only (domain check) | Collected with written consent; deleted after the project |

# 3. Team and roles

| Member | Roles |
|---|---|
| M1 – Dinh Phu Nguyen | Scrum Master (all sprints), group contact, ML engineer, edge/embedded engineer, data, web/backend, QA, documentation, MLOps & DevOps lead |
| M2 – Phuc Lam Nguyen | Research and data support |

Every task has an owner and a backup owner (the other member). Escalation path for problems: talk early → record and re-plan → formal notice and reallocation → escalate to the tutor with evidence.

# 4. Sprint overview

| Sprint | Weeks | Goal |
|---|---|---|
| S0 | Wk 4 | Plan the project and set up tools (Group Task 1) |
| S1 | Wk 5–6 | Data pipeline ready; Pi capturing frames |
| S2 | Wk 7–8 | Train, compare and select the model |
| S3 | Wk 9–10 | Integrate the end-to-end login flow |
| S4 | Wk 11–12 | Evaluate, monitor, document and demo |

# 5. Key design decisions

| ID | Decision | Reason |
|---|---|---|
| DD1 | Run the liveness check on the Raspberry Pi (edge), not in the cloud | Face images stay on the device; latency and resources are measured on the real target hardware |
| DD2 | Passive liveness first; blink challenge only as an optional second step | Passive detection covers print, screen and replay with no user effort; replayed videos can contain blinks |
| DD3 | Lightweight CNN (MiniFASNet / MobileNetV3-Small) exported to ONNX INT8; LBP+SVM as baseline | Under 5 MB, runs on the Pi CPU without an accelerator |
| DD4 | Subject-disjoint split, frozen test set, threshold chosen on validation | Prevents identity leakage and optimistic results |
| DD5 | Select by APCER at BPCER ≤ 5% plus Pi p95 latency, not accuracy | Classes are imbalanced (≈ 1:3); accepting an attack costs more than a retry |
| DD6 | Pi results are signed and bound to a server nonce; technical errors fail closed | Stops forged or replayed "pass" messages; failures never bypass the check |
| DD7 | No raw face images in logs; monitoring on aggregates; retraining only with consented data | Privacy requirement of the project specification |
| DD8 | GitHub + DVC + MLflow for versioning; GitHub Actions for CI/CD | Links code, data, experiments and models for reproducibility |

# 6. Targets (to be confirmed with the tutor)

- APCER minimised at BPCER ≤ 5% on validation; final results reported once on the frozen test set.
- Per-frame inference p95 ≤ 50 ms and full attempt p95 ≤ 2 s on the Raspberry Pi 5.
- Technical-failure rate < 1% during the demo test runs.

# 7. Glossary

| Term | Meaning |
|---|---|
| PAD | Presentation attack detection (anti-spoofing) |
| Bona fide | A genuine, live presentation |
| APCER | Attack presentation classification error rate – share of attacks wrongly accepted as live |
| BPCER | Bona fide presentation classification error rate – share of genuine users wrongly rejected |
| ACER | Average of APCER and BPCER |
| PSI | Population Stability Index – measures how much a distribution has shifted |

# 8. Meeting minutes

**Sprint 0 planning meeting – [DD/MM/2026], [Teams / in person]**

- Attendees: Dinh Phu Nguyen, Phuc Lam Nguyen
- Decisions: project scope (passive liveness on Pi 5), datasets (CelebA-Spoof main, Replay-Attack via tutor), tools (Jira, GitHub, DVC, MLflow), sprint plan S0–S4.
- Actions: [owner – action – due date]

# 9. Links

- GitHub repository: https://github.com/phu1009/COS40007---AI-Engineering-
- Jira board: [Jira board URL]
- Group Task 1 report: docs/task1/
