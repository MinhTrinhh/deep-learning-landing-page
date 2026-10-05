# Deep Learning and Its Applications (CO3133) - Course Project Repository

**Course:** CO3133 - Deep Learning and Its Applications (Semester-261)  
**Institution:** Ho Chi Minh City University of Technology (HCMUT), VNU-HCM  
**Faculty:** Faculty of Computer Science and Engineering  
**Instructor:** Lê Thành Sách  
**Group:** G-10

**Live GitHub Pages Site:** [https://MinhTrinhh.github.io/deep-learning-landing-page/docs/](https://MinhTrinhh.github.io/deep-learning-landing-page/docs/)

---

## Local Environment Setup

### Prerequisites

- Python 3.10 or later
- Git
- A terminal, or the integrated terminal in VS Code

Run the following commands from the repository root. Using a project-local virtual environment keeps this project's packages separate from the system Python installation. The `.venv/` directory is already excluded by `.gitignore`.

### 1. Clone and enter the repository

```bash
git clone https://github.com/MinhTrinhh/deep-learning-landing-page.git
cd deep-learning-landing-page
```

If the repository is already cloned, open its root directory in your terminal and continue with the next step.

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

After activation, the terminal prompt should normally include `(.venv)`.

### 3. Install the project dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs the package versions declared in the repository-level [`requirements.txt`](requirements.txt).

### 4. Select the environment in VS Code

1. Open the Command Palette with `Ctrl+Shift+P` or `Cmd+Shift+P`.
2. Run **Python: Select Interpreter**.
3. Select the interpreter inside `.venv`.

To leave the environment when finished:

```bash
deactivate
```

## Assignment-Specific Guidance

The root README only covers shared repository setup. For dataset preparation, supported commands, training, evaluation, generated artifacts, checkpoints, and reproducibility details, follow the README inside the relevant assignment directory.

- [Assignment 1 setup and usage](Assignment1/README.md)

Assignment-specific instructions for later assignments will be added to their respective directories as those implementations are completed.
