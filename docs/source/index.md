# Customer Support Ticket Classification

A almost-production-ready MLOps project for automatically classifying customer support ticket priority using DistilBERT.

## Overview

This project implements an end-to-end machine learning pipeline for classifying customer support tickets into three priority levels: **low**, **medium**, and **high**. It demonstrates MLOps best practices including experiment tracking, data versioning, containerized training, and cloud deployment.

## Architecture

```
Kaggle Dataset
      |
      v
Data Preprocessing (tokenization, splitting)
      |
      v
DVC (versioned data on GCS)
      |
      v
Training (Local or Vertex AI)
      |
      v
Experiment Tracking (Weights & Biases)
      |
      v
Model Checkpoint
      |
      v
FastAPI Inference Service
```

## Key Features

- **DistilBERT Model** - Multilingual transformer for text classification
- **PyTorch Lightning** - Scalable training with mixed precision and distributed support
- **Hydra Configuration** - Flexible experiment management with config composition
- **Weights & Biases** - Experiment tracking and model versioning
- **DVC** - Data versioning with Google Cloud Storage backend
- **FastAPI** - REST API for real-time inference
- **Docker** - Containerized training (CPU/GPU) and serving
- **Vertex AI** - Managed cloud training on GCP
- **GitHub Actions** - CI/CD with multi-platform testing

## Quick Links

| Topic | Description |
|-------|-------------|
| [Getting Started](getting_started.md) | Setup guide for new developers |
| [Data Processing](data.md) | Dataset details and preprocessing |
| [Model](model.md) | Model architecture and configuration |
| [Training](training.md) | Local and cloud training guide |
| [API](api.md) | Inference service documentation |
| [CLI Commands](cli.md) | All available invoke tasks |
| [Testing](testing.md) | Test suite and CI/CD |
| [DVC](dvc.md) | Data versioning workflow |
| [Cloud](cloud.md) | GCP infrastructure and Vertex AI |
