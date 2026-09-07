# Course AI Usage Disclosure Log

**Course:** CO3133 - Deep Learning and Its Applications (Semester-261)  
**Institution:** Ho Chi Minh City University of Technology (HCMUT), VNU-HCM  
**Faculty:** Faculty of Computer Science and Engineering  
**Group:** Group [ID]  
**Repository:** [https://github.com/your-username/deep-learning-repo](https://github.com/your-username/deep-learning-repo)

---

## Policy Statement

In accordance with Section 5 of the Course Project Handbook, all generative AI tool usage across Assignments 1–3 is transparently disclosed, traceable, and verified. No unchecked AI outputs are submitted, and all experimental results, performance numbers, and figures are produced from actual code execution.

---

## AI Usage Disclosure Log Table

| Entry ID | Tool & Model Name | Used By | Development Stage | Purpose & Task Description | Affected Files / Sections | Student Verification Method & Source | Responsible Member |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LOG-001` | Antigravity AI (Gemini 3.6 Flash) | Member 1 | GitHub Pages Skeleton Setup | Created initial HTML/CSS responsive site skeleton for landing page and assignments | `docs/index.html`, `Assignment1-3/index.html`, `styles.css` | Manually verified DOM accessibility, links, CSS rules, and handbook compliance | Member 1 |
| `LOG-002` | ChatGPT (GPT-4o) | Member 3 | Assignment 1 Model Implementation | Prompted for PyTorch Vision Transformer patch projection tensor reshapes | `Assignment1/code/models/vit.py`, Assignment 1 Report Sec 3.2 | Tested shape compatibility with PyTorch official ViT documentation and unit tests | Member 3 |

---

## Detailed Log Entries

### Entry `LOG-001`
- **Tool:** Antigravity AI (Gemini 3.6 Flash)
- **Used by:** Member 1
- **Task:** Creating GitHub Pages Skeleton
- **Prompt Summary:** Requested handbook-compliant GitHub Pages skeleton structure for CO3133.
- **AI Contribution:** Produced responsive HTML files, CSS variables design system, and JS tab switcher matching the architectural diagram.
- **Student Verification:** Tested layout on desktop and mobile viewports, verified all handbook required sections.
- **Affected Files:** `docs/index.html`, `Assignment1/index.html`, `Assignment2/index.html`, `Assignment3/index.html`, `docs/assets/css/styles.css`
- **Responsible Member:** Member 1

---

### Entry `LOG-002`
- **Tool:** ChatGPT (GPT-4o)
- **Used by:** Member 3
- **Task:** Vision Transformer Patch Embedding Layer
- **Prompt Summary:** Asked how to convert `[B, C, H, W]` image tensors into `[B, N, D]` patch tokens in PyTorch.
- **AI Contribution:** Suggested using `nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)` followed by `flatten(2).transpose(1, 2)`.
- **Student Verification:** Tested tensor shapes in Jupyter notebook and verified against PyTorch documentation.
- **Affected Files:** `Assignment1/code/models/vit.py`, Assignment 1 Report Sec 3.2
- **Responsible Member:** Member 3

---

## Declaration of Non-AI Components (If Applicable)

*If any assignment or section was completed without any generative AI assistance, state:*
> The group explicitly declares that no generative AI tool was used in [Assignment X / Section Y].
