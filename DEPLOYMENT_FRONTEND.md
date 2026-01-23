# Deploying Streamlit Frontend to Google Cloud

This guide covers deploying the Customer Support Ticket Classifier Streamlit app to Google Cloud Run.

## Prerequisites

- Google Cloud Project set up
- `gcloud` CLI installed and configured
- Artifact Registry API enabled
- Cloud Run API enabled

## Quick Start

### 1. Set up environment variables

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"
export REGISTRY="${REGION}-docker.pkg.dev"
```

### 2. Create Artifact Registry repository (if not exists)

```bash
gcloud artifacts repositories create mlops-registry \
  --repository-format=docker \
  --location=$REGION \
  --project=$PROJECT_ID
```

### 3. Build and push Docker image

**Option A: Using Cloud Build (recommended for CI/CD)**

```bash
gcloud builds submit \
  --config=cloudbuild-frontend.yaml \
  --project=$PROJECT_ID \
  --substitutions=_REGION=$REGION
```

**Option B: Build locally**

```bash
# Authenticate Docker with Artifact Registry
gcloud auth configure-docker ${REGISTRY}

# Build image
docker build -t ${REGISTRY}/${PROJECT_ID}/mlops-registry/frontend:latest \
  -f dockerfiles/frontend.dockerfile .

# Push image
docker push ${REGISTRY}/${PROJECT_ID}/mlops-registry/frontend:latest
```

### 4. Deploy to Cloud Run

```bash
gcloud run deploy customer-support-classifier \
  --image ${REGISTRY}/${PROJECT_ID}/mlops-registry/frontend:latest \
  --platform managed \
  --region $REGION \
  --project $PROJECT_ID \
  --memory 2Gi \
  --cpu 1 \
  --timeout 3600 \
  --allow-unauthenticated
```

### 5. Get the service URL

```bash
gcloud run services describe customer-support-classifier \
  --region $REGION \
  --project $PROJECT_ID \
  --format='value(status.url)'
```

Visit the URL to access your Streamlit app!

## Configuration

### Environment Variables

Set these when deploying:

```bash
gcloud run deploy customer-support-classifier \
  --set-env-vars MODEL_PATH=/app/models/model.ckpt \
  # ... other flags
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/model.ckpt` | Path to model checkpoint (demo mode if not found) |
| `STREAMLIT_SERVER_PORT` | `8080` | Port for Streamlit server |

### Using a Model Checkpoint

If you have a trained model:

1. **Upload to Google Cloud Storage:**

```bash
gsutil cp models/model.ckpt gs://${PROJECT_ID}-models/model.ckpt
```

2. **Mount in Cloud Run (via Cloud Build with volumes)** or use **Cloud Storage FUSE**:

```bash
gcloud run deploy customer-support-classifier \
  --image ${REGISTRY}/${PROJECT_ID}/mlops-registry/frontend:latest \
  --set-env-vars MODEL_PATH=/gcs/${PROJECT_ID}-models/model.ckpt \
  # ... other flags
```

## Scaling

Adjust resources based on traffic:

```bash
gcloud run deploy customer-support-classifier \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 100 \
  # ... other flags
```

## Monitoring

### View logs

```bash
gcloud run logs read customer-support-classifier \
  --region $REGION \
  --project $PROJECT_ID
```

### View metrics

```bash
gcloud monitoring dashboards list \
  --project $PROJECT_ID
```

## Demo Mode

If no model checkpoint is provided, the app runs in **demo mode**:
- Shows a warning banner
- Uses keyword-based priority classification
- Perfect for testing UI without ML infrastructure

## Cost Optimization

- **Reduce memory**: Start with 1GB if model is small
- **Set min instances**: 0 for auto-scaling (cold starts ~30s)
- **Use Cloud Build caching**: Speeds up rebuilds
- **Monitor logs**: Identify and fix inefficiencies

## Troubleshooting

### App times out during startup

Increase timeout and memory:

```bash
gcloud run deploy customer-support-classifier \
  --timeout 3600 \
  --memory 4Gi
```

### Model not found in demo mode

The app should show a warning. Either:
1. Set `MODEL_PATH` to an existing checkpoint
2. Upload model to Cloud Storage and use `gs://` path

### Container image size too large

Use multi-stage builds (already included in Dockerfile) and remove unnecessary dependencies.

## CI/CD Integration

The `cloudbuild-frontend.yaml` automatically:
1. Builds Docker image on git push
2. Pushes to Artifact Registry
3. Deploys to Cloud Run

Trigger builds with:

```bash
gcloud builds triggers create github \
  --name=frontend-ci \
  --repo-name=mlops-project \
  --repo-owner=your-username \
  --branch-pattern="^main$" \
  --build-config=cloudbuild-frontend.yaml
```

## Next Steps

- Add authentication (Firebase, OAuth)
- Set up custom domain with Cloud CDN
- Configure alerts for errors and high latency
- Monitor costs with Cloud Billing

For more info: [Cloud Run Documentation](https://cloud.google.com/run/docs)
