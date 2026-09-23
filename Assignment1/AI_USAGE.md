# AI Usage Disclosure

**Course:** CO3133 — Deep Learning and Its Applications · Semester-261
**Group:** G-10
**Assignment:** Assignment 1 — Foundations of Deep Learning Pipelines and Architectures

This file logs all generative AI tool usage for this assignment, per the course
Academic Integrity and AI-Use Policy. If no AI was used, delete the log entries below
and keep only the declaration in Section 2.

---

## Detailed Log

```
Tool: Gemini
Used by: Trịnh Lê Minh
Task: Implementing model checkpointing and analyzing loss curves
Prompt summary: Asked for the definition and code implementation of machine learning checkpoints, and requested an analysis of a loss/accuracy graph to determine if the model was overfitting.
AI contribution: Provided the PyTorch code logic to save and load best-model checkpoints based on validation loss. Analyzed the provided training/validation graphs, confirming healthy training and explaining how the MLP's Dropout and Data Augmentation caused validation metrics to initially outperform training metrics.
Student verification: Verified the health of the model's training/validation loss curve to confirm no overfitting was occurring; reviewed and integrated the PyTorch checkpointing code logic into the training loop.
Affected files/sections: trainer.py
Responsible member: Trịnh Lê Minh

Tool: OpenAI Codex (GPT-5)
Used by: Trịnh Lê Minh
Task: Building and debugging a reproducible FashionMNIST training and evaluation pipeline
Prompt summary: Asked for guidance and implementation support for EDA, train/validation/test splitting, preprocessing and augmentation, Linear/MLP training, evaluation metrics, checkpointing, and reproducibility. Later reported a PyTorch checkpoint-loading error caused by NumPy scalar metadata.
AI contribution: Connected the EDA and metrics modules to the main workflow; added saved EDA figures, classification metrics, confusion matrices, best-validation-loss checkpoints, deterministic seeding, DataLoader worker seeding, experiment metadata, and model fingerprints. Diagnosed the checkpoint error, converted checkpoint metadata to native Python types, and added a restricted compatibility loader for the existing checkpoint.
Student verification: Ran the MLP training workflow and supplied the resulting checkpoint-loading traceback; inspected the generated metric files and checkpoints. Existing best_mlp.pt was subsequently loaded successfully, and syntax/static checks passed.
Affected files/sections: Assignment1/code/main.py, config.py, dataloader.py, edaworker.py, metrics.py, trainer.py, plotter.py; Assignment1/code/README.md; Assignment1/checkpoints/README.md.
Responsible member: Trịnh Lê Minh

Tool: GPT (via VSCode)
Used by: Tran Phuoc Sang
Task: Coding assistance for metrics and plotter files — task-specific EDA
Prompt summary: Asked for help implementing metric computation and EDA plot functions
AI contribution: Suggested structure for metric aggregation and matplotlib plot helpers
Student verification: Ran outputs against expected values; visually verified EDA plots
Affected files/sections: Assignment1/code/metrics.py; Assignment1/code/edaworker.py
Responsible member: Tran Phuoc Sang

Tool: Claude Sonnet 4.6 (Antigravity IDE)
Used by: Tran Phuoc Sang
Task: LaTeX report writing assistance
Prompt summary: Asked for help drafting methodology and results sections in LaTeX
AI contribution: Suggested LaTeX structure, table formatting, phrasing for Sec. 3–4
Student verification: Cross-checked all claims against actual experiment outputs and code
Affected files/sections: report/report-assignment-1/ (Methodology, Results sections)
Responsible member: Tran Phuoc Sang
```
---

## Student Responsibility Acknowledgement

- [x] All members understand and can explain every part of the submission.
- [x] No unchecked AI-generated code or text was submitted as-is.
- [x] AI was not used to fabricate data, experimental results, citations, or references.
- [x] No experiments were claimed that were not actually run.
- [x] No private/restricted/credential-bearing data was pasted into AI tools.
- [x] All AI-assisted content was verified via source code, real runs, official docs, or scholarly sources.

---

> **Consistency note:** The AI usage entries above are consistent with the disclosure in
> [`docs/index.html`](../docs/index.html) (landing page),
> the AI Disclosure tab in [`Assignment1/index.html`](index.html), and
> the `AI_Usage_Disclosure.tex` section of the Assignment 1 LaTeX report.
