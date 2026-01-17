# Customer IT Support - Ticket Classification

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-yellow)
![W&B](https://img.shields.io/badge/W%26B-FFCC33?logo=weightsandbiases&logoColor=black)
![DVC](https://img.shields.io/badge/DVC-945DD6?logo=dvc&logoColor=white)

[![Tests](https://github.com/DTU-MLOps-Group-41/mlops-project/actions/workflows/tests.yaml/badge.svg)](https://github.com/DTU-MLOps-Group-41/mlops-project/actions/workflows/tests.yaml)
[![Codecov](https://codecov.io/github/DTU-MLOps-Group-41/mlops-project/graph/badge.svg?token=2896K468ZM)](https://codecov.io/github/DTU-MLOps-Group-41/mlops-project)
[![Linting](https://github.com/DTU-MLOps-Group-41/mlops-project/actions/workflows/linting.yaml/badge.svg)](https://github.com/DTU-MLOps-Group-41/mlops-project/actions/workflows/linting.yaml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)
![Renovate](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)

## Group 41 — DTU MLOps Course Project

- **Sanjit Srinivasan** (s256657)
- **João Prazeres** (s243036)
- **Kornel Gładkowski** (s242908)
- **Mounika Maidamshetti** (s250148)


This repository contains the project work carried out by group 41 in the MLOps course taught at DTU ([course website](https://skaftenicki.github.io/dtu_mlops/)).

###  Overall goal of the project

The goal of the project is to apply natural language processing (NLP) techniques to a supervised text classification problem, namely the automatic prediction of the priority level of customer support tickets and, optionally, the subject or department to which each ticket should be routed. The main goal of this project is developing and deploying the solution in a streamlined, reproducible, and efficient manner that reflects real-world machine learning workflows.

###  Frameworks used

We are using the [PyTorch](https://pytorch.org) and [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index) framework. [Weights & Biases](https://wandb.ai/) for experiment tracking and model registry, [DVC](https://dvc.org/) for data versioning, FastAPI for serving; these frameworks are core components of the system and are fully integrated into the project lifecycle.

### Data

We are using the [Kaggle](https://www.kaggle.com/) dataset [Customer IT Support - Ticket Dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets). This dataset contains a total of 28,600 observations, each of which will have a topic assigned by the customer, a description of the customer's issue or inquiry, and a priority level (low, medium, or critical) assigned to the ticket. Additionally, each observation will have a department to which the email ticket is categorized.

###  Models

We intend to use a pre-trained natural language processing (NLP) model. To train the model and perform hyperparameter sweeping, we will initially use compressed versions of [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert), such as [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert) or [ALBERT](https://huggingface.co/docs/transformers/en/model_doc/albert). These compressed versions allow for more efficient training and a greater focus on the MLOps aspect of the project.

## Project structure

The directory structure of the project looks like this:
```txt
.
├── configs                   # Configuration files
├── .devcontainer             # Development container configuration
│   ├── devcontainer.json
│   └── post_create.sh
├── dockerfiles               # Dockerfiles
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs                      # Documentation
│   ├── source
│   │   └── index.md
│   ├── mkdocs.yaml
│   └── README.md
├── .github                   # GitHub actions and automation
│   ├── agents
│   │   └── dtu_mlops_agent.md
│   ├── prompts
│   │   └── add_test.prompt.md
│   └── workflows
│       ├── linting.yaml
│       ├── pre-commit-update.yaml
│       └── tests.yaml
├── models                    # Trained models
├── notebooks                 # Jupyter notebooks
├── reports                   # Reports
│   └── figures
├── src                       # Source code
│   └── customer_support
│       ├── api.py
│       ├── data.py
│       ├── evaluate.py
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       └── visualize.py
├── tests                     # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── AGENTS.md
├── .gitignore
├── LICENSE
├── .pre-commit-config.yaml
├── pyproject.toml            # Python project file
├── .python-version
├── README.md                 # Project README
├── renovate.json             # Renovate configuration
├── tasks.py                  # Project tasks
└── uv.lock                   # UV lock file
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
