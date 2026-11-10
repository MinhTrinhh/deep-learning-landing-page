# Assignment 2 — Deep Learning on Large-Scale Data and Specialized Tasks

Deep Learning on Large-Scale Data for Specialized Computer Vision or Language Tasks and evaluation using a custom BiLSTM model and a fine-tuned DistilledBERT model 
on downstream tasks

- [Assignment 2 website](https://MinhTrinhh.github.io/deep-learning-landing-page/Assignmentw/)
- [AI usage disclosure](AI_USAGE.md)

## Directory Structure

```text
Assignment2/
├── AI_USAGE.md              # Assignment-specific AI disclosure
├── README.md                # Installation and execution guide
├── index.html               # Assignment website
├── code/
│   ├── EDA.py               # EDA calculations
│   ├── main.py              # Command-line entry point
│   ├── metrics.py           # Predictive and resource metrics
│   ├── baseline_model.py    # Baseline BiLSTM architecture for evaluation
│   ├── plotter.py           # EDA, learning-curve, and metric plots
│   ├── qualitative.py       # Reproducible qualitative checkpoint comparison
│   ├── dataset.py           # Dataset class, split, and loaders 
├── dataset/                 # Corona_NLP dataset
├── diagram-pipeline/        # Methodology diagram assets
├── training_report/         # Log files on the most recent training runs
└── outputs/
    ├── eda/                 # EDA figures
    ├── learning_curves/     # Training and validation curves
    └── metrics/             # Confusion-matrix figures
```

## Installation

Create and activate the virtual environment by following the [repository setup guide](../README.md#local-environment-setup). Then install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Move into the Assignment 2 code directory before running the commands below:

```bash
cd Assignment2/code
```

## Dataset Preparation

Download the dataset from this source: [Corona_NLP](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data) then extract the contents into `Assignment2/dataset`

To generate the EDA outputs before training all implemented models:

```bash
python EDA.py
```

## Training Commands

Train the Baseline model:

```bash
python code/main.py --base
```

Fine-tune the Pretrained model:

```bash
python code/main.py --pretrained
```

Fine-tune the Pretrained model while freezing the backbone for ablation:

```bash
python code/main.py --pretrained --freeze
```

## Evaluation Commands

For training, evaluation is automatic after every training command. The trainer restores the checkpoint with the lowest F1 score, evaluates it on the official test set, prints the metrics, and writes the generated artifacts under `Assignment2/outputs/`.

For standalone evaluation, use the `--eval` tag after the model name, for example:

```bash
python code/main.py --pretrained --eval
```

This loads the best checkpoint of the corresponding model and performs evaluation alongside plotting of the results.

## Obtaining Checkpoints

The checkpoints to the models found in this analysis can be found at [Google Drive](https://drive.google.com/file/d/1HIwmfVmxGnuoOsVf7wNrxPjkYVyFS6e0/view?usp=sharing), download and extract it in `Assignment2` so that the final destination is `Assignment2/checkpoints` 

## Qualitative evaluation Commands

Generate the qualitative comparison used in the report after all
checkpoints are available:

```bash
python code/main.py --pretrained --infer
```

OR:

```bash
python code/main.py --base --infer
```
