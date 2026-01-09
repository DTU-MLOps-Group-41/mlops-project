# customer_support

## Group 41 — DTU MLOps Course Project

- **Sanjit Srinivasan** (s256657)
- **João Prazeres** (s243036)
- **Kornel Gładkowski** (s242908)
- **Mounika Maidamshetti** (s250148)


This repository contains the project work carried out by group 41 in the MLOps course taught at DTU (course website). 
###  Overall goal of the project
The goal of the project is to apply natural language processing (NLP) techniques to a supervised text classification problem, namely the automatic prediction of the priority level of customer support tickets and, optionally, the subject or department to which each ticket should be routed. The main goal of this project is developing and deploying the solution in a streamlined, reproducible, and efficient manner that reflects real-world machine learning workflows.

###  Frameworks used
We are using the PyTorch and Hugging Face Transformers framework. Weights & Biases for experiment tracking and model registry, DVC for data versioning, FastAPI for serving; these frameworks are core components of the system and are fully integrated into the project lifecycle

### Data
We are using the Kaggle dataset Customer IT Support - Ticket Dataset. This dataset contains a total of 28,600 observations, each of which will have a topic assigned by the customer, a description of the customer's issue or inquiry, and a priority level (low, medium, or critical) assigned to the ticket. Additionally, each observation will have a department to which the email ticket is categorized. 

###  Models
We intend to use a pre-trained natural language processing (NLP) model. To train the model and perform hyperparameter sweeping, we will initially use compressed versions of BERT, such as DistilBERT (DistilBERT) or ALBERT (ALBERT). These compressed versions allow for more efficient training and a greater focus on the MLOps aspect of the project.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
