## Cloud Infrastructure

This project uses Google Cloud Platform (GCP) for cloud-based training. The infrastructure includes Cloud Build for CI/CD, Artifact Registry for Docker images, Vertex AI for managed training, and Google Cloud Storage for data (via DVC).

## Architecture Overview

```
GitHub Repository
        │
        │ push to main
        ▼
┌───────────────────┐
│   Cloud Build     │ ◄── Automatic trigger
│  (Docker images)  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Artifact Registry │
│  europe-west1     │
└─────────┬─────────┘
          │
          │ Manual: gcloud builds submit
          ▼
┌───────────────────┐      ┌───────────────┐
│    Vertex AI      │ ◄────│  GCS Bucket   │
│  Training Jobs    │      │  (DVC data)   │
└─────────┬─────────┘      └───────────────┘
          │
          ▼
┌───────────────────┐
│ Weights & Biases  │
│ (Experiment logs) │
└───────────────────┘
```

**Key Components:**

- **Artifact Registry**: `europe-west1-docker.pkg.dev/corded-smithy-484309-s4/g41-registry/`
- **GCS Data Bucket**: `gs://my_mlops_data_bucket3/`
- **WandB**: Entity `dtu-mlops-g41`, Project `customer_support`
- **Region**: `europe-west1`

## Docker Images

Two training images are maintained:

| Image | Base | Use Case |
|-------|------|----------|
| `train-cpu` | `uv:python3.12-bookworm-slim` | CPU-only training |
| `train-cu128` | `nvidia/cuda:12.8.0-runtime-ubuntu24.04` | GPU training (CUDA 12.8) |

### Automatic Builds

Docker images are automatically built and pushed when commits are pushed to the `main` branch. This is configured as a Cloud Build trigger in the GCP Console.

The build process (`.gcp/cloudbuild.yaml`) uses Docker BuildKit and pushes to Artifact Registry.

### Manual Image Build

To manually trigger a build:

```bash
# Build GPU image
gcloud builds submit . --config=.gcp/cloudbuild.yaml \
  --substitutions=_DEVICE=cu128,_IMAGE_NAME=train-cu128:latest,_DOCKERFILE=dockerfiles/train_cu128.dockerfile

# Build CPU image
gcloud builds submit . --config=.gcp/cloudbuild.yaml \
  --substitutions=_DEVICE=cpu,_IMAGE_NAME=train-cpu:latest,_DOCKERFILE=dockerfiles/train_cpu.dockerfile
```

### Container Entrypoint

Both images use `dockerfiles/entrypoint.sh` which:

1. Configures DVC for containerized environment (`core.no_scm true`)
2. Pulls training data from GCS via DVC
3. Runs the training script with provided arguments

## Vertex AI Training

### Submitting Training Jobs

Training jobs are submitted via Cloud Build, which handles secret injection and job creation:

```bash
# GPU training (recommended for production)
gcloud builds submit . --config=.gcp/vertex_ai_train.yaml \
  --substitutions=_VERTEX_TRAIN_CONFIG=.gcp/config_gpu.yaml

# CPU training (for testing)
gcloud builds submit . --config=.gcp/vertex_ai_train.yaml \
  --substitutions=_VERTEX_TRAIN_CONFIG=.gcp/config_cpu.yaml
```

### Training Configurations

**GPU Configuration** (`.gcp/config_gpu.yaml`):

- Machine: `n1-standard-8` with 1x NVIDIA Tesla T4
- Image: `train-cu128:latest`
- Default experiment: `experiment=best` (optimized hyperparameters, full dataset)

**CPU Configuration** (`.gcp/config_cpu.yaml`):

- Machine: `n1-standard-8`
- Image: `train-cpu:latest`
- Default experiment: `experiment=baseline` (quick testing, small dataset)

### Customizing Training

Modify the `args` in the config files to pass Hydra overrides:

```yaml
args:
  - "experiment=best"
  - "training.num_epochs=10"
  - "training.batch_size=32"
```

Available experiments:

- `experiment=baseline` - Quick testing on small dataset
- `experiment=best` - Optimized hyperparameters on full dataset

## Secrets Management

The WandB API key is stored in Google Secret Manager and automatically injected during training job submission.

## Experiment Tracking

Training runs are logged to Weights & Biases:

- **Entity**: `dtu-mlops-g41`
- **Project**: `customer_support`
- **Dashboard**: [wandb.ai/dtu-mlops-g41/customer_support](https://wandb.ai/dtu-mlops-g41/customer_support)

Configuration in `configs/config.yaml`:

```yaml
wandb:
  entity: "dtu-mlops-g41"
  project: "customer_support"
  mode: "online"
  log_model: true
```

**Note**: WandB Model Registry integration is not yet configured. Currently `log_model: true` logs model artifacts to individual runs.

## Prerequisites

### GCP Authentication

```bash
gcloud auth login
gcloud config set project corded-smithy-484309-s4
gcloud config set compute/region europe-west1
```

### Required Permissions

- Cloud Build Editor - submit builds
- Vertex AI User - create training jobs
- Storage Object Viewer - access DVC data
- Secret Manager Secret Accessor - access WANDB_API_KEY (for service account)

## Monitoring

### Cloud Build

```bash
# List recent builds
gcloud builds list --limit=10

# Stream build logs
gcloud builds log BUILD_ID --stream
```

### Vertex AI Jobs

```bash
# List training jobs
gcloud ai custom-jobs list --region=europe-west1

# View job details
gcloud ai custom-jobs describe JOB_ID --region=europe-west1
```

## Troubleshooting

**Build fails with permission errors**

- Verify Cloud Build service account has Artifact Registry Writer role
- Check Secret Manager access for WANDB_API_KEY

**Training job fails to start**

- Check GPU quota in europe-west1 region
- Verify image exists: `gcloud artifacts docker images list europe-west1-docker.pkg.dev/corded-smithy-484309-s4/g41-registry`

**DVC pull fails in container**

- Ensure GCS bucket allows access from Vertex AI service account
- Check `.dvc/config` remote URL matches bucket
