# AI Usage Disclosure

**Course:** CO3133 — Deep Learning and Its Applications · Semester-261
**Group:** G-10
**Assignment:** Assignment 2 — Deep Learning on Large-Scale Data and Specialized Tasks - Specialized Computer Vision / Language Track

This file logs all generative AI tool usage for Assignment 2.

---

## Detailed Log

```
Tool: Claude Sonnet 5.5 (Anthropic, claude.ai chat)
Used by: Đặng Trần Thái Bảo
Task: Adding an evaluation-only mode (--eval)  to main.py
Prompt summary: Asked to modularize evaluation feature to a single --eval command that can be triggered directly on the terminal, skipping the training process. Evaluation pipeline had already been present within the code, AI was only responsible to move and rewire these features into a standalone module.
Affected files/sections: Assignment2/code/main.py.
Responsible member: Đặng Trần Thái Bảo

Tool: Claude Sonnet 5.5 (Anthropic, claude.ai chat)
Used by: Đặng Trần Thái Bảo
Task: Converting EDA writings into web page format
Prompt summary: Asked the AI to convert existing writings into html format for web page representation while fact-checking and rounding up ideas 
AI contribution: Wrote the "Representative Samples" section from the supplied figure, added the table of duplicated training tweets to the duplicate-check section, and made a small edit to the EDA introduction.
Student verification: Supplied the writing contents, figure and the duplicate-tweet list generated from the dataset
Affected files/sections: Assignment2/index.html (EDA tab).
Responsible member: Đặng Trần Thái Bảo

Tool: Claude Sonnet 5.5 (Anthropic, claude.ai chat)
Used by: Đặng Trần Thái Bảo
Task: Converting Methodology writings into web page format
Prompt summary: Asked the AI to convert existing writings into html format for web page representation while fact-checking and rounding up ideas 
AI contribution: Calculated the DistilBERT head, frozen-backbone and BiLSTM parameter counts; rewrote the frozen-backbone description to the controlled setting (same learning rate and epoch budget as full fine-tuning).
Student verification: Supplied the writing ideas, contents, dataset.py, baseline_model.py and main.py so each value could be checked against the code, and supplied the hardware details and the experiment decisions.
Affected files/sections: Assignment2/index.html (Methodology tab); diagram paths under Assignment2/diagram-pipeline/ and Assignment2/diagram-models/.
Responsible member: Đặng Trần Thái Bảo

Tool: Claude Sonnet 5.5 (Anthropic, claude.ai chat)
Used by: Đặng Trần Thái Bảo
Task: Converting Results writings into web page format
Prompt summary: Supplied the final test output of the three experiments, then the per-epoch training logs and confusion matrices, and asked for the results tab to be filled, the frozen-backbone results to be updated;
AI contribution: Filled the headline, detailed-metric, per-class and resource tables; checked derived figures (differences, ratios, validation-to-test gaps) with scripts; per-class and accuracy-versus-cost observations; computed error-structure statistics from the confusion matrices; wrote the limitations and conclusion; and flagged that the higher-learning-rate frozen run broke the single-factor comparison.
Student verification: Ran all experiments and supplied the ideas, writings, console logs, confusion matrices and figures; decided to revert the frozen run to the shared 2e-5 / 5-epoch setting and re-ran it.
Affected files/sections: Assignment2/index.html (Results tab).
Responsible member: Đặng Trần Thái Bảo

Tool: Claude Sonnet 5.5 (Anthropic, claude.ai chat)
Used by: Đặng Trần Thái Bảo
Task: Error analysis and compute-cost analysis for the Assignment 2 web page
Prompt summary: Supplied with ideas on failure-categoies, asked for implementation into html and ideas round-up, qualitative failure cases and a hardware compute-utilization analysis (GPU hours, memory footprint); supplied the qualitative prediction figure and the GPU model.
AI contribution: Defined a three-category error taxonomy (intensity errors, neutral-boundary errors, polarity flips) and computed the category counts, direction-of-error and most-common-mistake tables from the confusion matrices; described six failure cases from the supplied qualitative_predictions.png and a separate label-ambiguity table; calculated GPU-hours, time to the best checkpoint, throughput, effective FLOP rate.
Student verification: Produced the qualitative figure with the --infer mode and supplied it, along with the hardware model; memory figures are stated on the page as analytic estimates, not measurements.
Affected files/sections: Assignment2/index.html (Error Analysis & Compute Cost tab).
Responsible member: Đặng Trần Thái Bảo

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