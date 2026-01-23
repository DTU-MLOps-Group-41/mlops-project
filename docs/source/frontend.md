# Frontend Service

The project includes a Streamlit web interface for interactive ticket priority classification with integration to the FastAPI backend.

## Overview

The frontend is a user-friendly web application that classifies customer support tickets in real-time:

- **Framework**: Streamlit
- **Backend**: FastAPI (optional, with local model fallback)
- **Port**: 8501 (default)
- **Model**: DistilBERT with local inference support

## Quick Start

### Running Locally

**Terminal 1: Start FastAPI Backend**

```bash
uv run uvicorn customer_support.api:app --host 0.0.0.0 --port 8080 --reload
```

**Terminal 2: Start Streamlit Frontend**

```bash
uv run invoke frontend
```

Or directly:

```bash
uv run streamlit run src/customer_support/frontend.py
```

The frontend opens at `http://localhost:8501`

## Architecture

```
User Browser (Port 8501)
    ↓
Streamlit UI
    ↓
    ├─→ Try HTTP POST /predict to API (Port 8080)
    │
    └─→ Fallback: Local DistilBERT Model
```

## Features

### Interactive UI
- **Text Input**: Paste or type customer support tickets
- **Priority Classification**: Displays predicted priority (Low, Medium, High)
- **Confidence Score**: Shows model confidence as percentage
- **Prediction History**: Tracks all predictions in current session
- **Color-Coded Badges**: Visual indicators for priority levels
  - 🟢 Low (Green)
  - 🟡 Medium (Orange)
  - 🔴 High (Red)

### Configuration
- **API URL Setting**: Configure backend URL in sidebar
- **Connection Testing**: Built-in button to test API connectivity
- **Environment Variables**: Support for `API_URL` env var

### Inference Modes

| Mode | Description | When Used |
|------|-------------|-----------|
| **API** (🌐) | Calls FastAPI backend | Primary mode, fastest |
| **Local** (🖥️) | Uses local DistilBERT model | API unavailable, offline |
| **Demo** (🎯) | Keyword-based classification | Model not found |

## Configuration

### Environment Variables

```bash
# Set custom API URL (default: http://localhost:8080)
export API_URL=http://api.example.com:8080

# Set custom model path (default: models/model.ckpt)
export MODEL_PATH=/path/to/model.ckpt

# Start frontend
uv run invoke frontend
```

### Sidebar Options

- **FastAPI URL**: Change API endpoint in real-time
- **Check API Connection**: Test if backend is reachable
- Shows current inference mode and API status

## Deployment to Google Cloud

### Prerequisites

- Google Cloud Project set up
- `gcloud` CLI installed and configured
- Artifact Registry API enabled
- Cloud Run API enabled

### Environment Setup

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"
export REGISTRY="${REGION}-docker.pkg.dev"
```

### Create Artifact Registry Repository

```bash
gcloud artifacts repositories create mlops-registry \
  --repository-format=docker \
  --location=$REGION \
  --project=$PROJECT_ID
```

### Build and Push Docker Image

**Option A: Using Cloud Build (recommended for CI/CD)**

```bash
gcloud builds submit \
  --config=cloudbuild-frontend.yaml \
  --project=$PROJECT_ID \
  --substitutions=_REGION=$REGION
```

**Option B: Build Locally**

```bash
# Authenticate Docker
gcloud auth configure-docker ${REGISTRY}

# Build image
docker build -t ${REGISTRY}/${PROJECT_ID}/mlops-registry/frontend:latest \
  -f dockerfiles/frontend.dockerfile .

# Push image
docker push ${REGISTRY}/${PROJECT_ID}/mlops-registry/frontend:latest
```

### Deploy to Cloud Run

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

### Get Service URL

```bash
gcloud run services describe customer-support-classifier \
  --region $REGION \
  --project $PROJECT_ID \
  --format='value(status.url)'
```

## Deployment Configuration

### Environment Variables on Cloud Run

```bash
gcloud run deploy customer-support-classifier \
  --set-env-vars MODEL_PATH=/app/models/model.ckpt \
  --set-env-vars API_URL=https://api-xxxxx.run.app \
  # ... other flags
```

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://localhost:8080` | FastAPI backend URL |
| `MODEL_PATH` | `models/model.ckpt` | Local model checkpoint path |
| `STREAMLIT_SERVER_PORT` | `8080` | Server port |

### Using a Model Checkpoint

**1. Upload to Google Cloud Storage:**

```bash
gsutil cp models/model.ckpt gs://${PROJECT_ID}-models/model.ckpt
```

**2. Deploy with Cloud Storage path:**

```bash
gcloud run deploy customer-support-classifier \
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

### Resource Guidelines

| Load | Memory | CPU | Min Instances |
|------|--------|-----|---------------|
| Low | 1Gi | 1 | 0 |
| Medium | 2Gi | 1 | 1 |
| High | 4Gi | 2 | 2 |

## Monitoring

### View Logs

```bash
gcloud run logs read customer-support-classifier \
  --region $REGION \
  --project $PROJECT_ID
```

### View Metrics

```bash
gcloud monitoring dashboards list --project $PROJECT_ID
```

## Demo Mode

If no model checkpoint is provided, the app runs in **demo mode**:
- Shows a warning banner
- Uses keyword-based priority classification
- Perfect for testing UI without ML infrastructure

### Demo Keywords

```
High Priority: urgent, critical, emergency, down, broken, crash, fail
Medium Priority: help, issue, problem, error, not working, unable
Low Priority: question, info, fyi, documentation, feature request
```

## Troubleshooting

### App Fails to Start

1. **Check dependencies:**
   ```bash
   uv run pip list | grep streamlit
   ```

2. **Check port availability:**
   ```bash
   lsof -i :8501
   ```

3. **Increase memory if using model:**
   ```bash
   gcloud run deploy ... --memory 4Gi
   ```

### API Connection Issues

1. **Check API is running:**
   ```bash
   curl http://localhost:8080/health
   ```

2. **Check API URL in sidebar** - Default is `http://localhost:8080`

3. **Check model checkpoint exists:**
   ```bash
   ls models/model.ckpt
   ```

4. **Check firewall rules** - May need to allow ports 8080 and 8501

### Slow Predictions

- Increase timeout: API has 30-second timeout
- Check CPU/GPU usage: Monitor resource utilization
- Check model size: Larger models take longer to load

### Container Image Too Large

The Dockerfile uses multi-stage builds to minimize image size. Verify in logs.

## Performance Tips

1. **API Response Time**: ~100-500ms (depends on GPU/CPU)
2. **Streamlit Caching**: Frontend caches responses to avoid duplicate calls
3. **Model Loading**: Model loads once at startup (cached in memory)
4. **Batch Predictions**: Consider batch API endpoint for many predictions

## Development Workflow

```bash
# 1. Make changes to frontend
vim src/customer_support/frontend.py

# 2. Streamlit auto-reloads on save (in demo mode)

# 3. Test predictions

# 4. Make model changes
vim src/customer_support/model.py

# 5. Retrain and update checkpoint
uv run invoke train

# 6. Restart API backend
# (Ctrl+C in Terminal 1, then rerun)

# 7. FastAPI auto-reloads with --reload flag
```

## Docker Compose (Local Testing)

```yaml
version: '3'
services:
  api:
    build:
      context: .
      dockerfile: dockerfiles/api.dockerfile
    ports:
      - "8080:8080"
    environment:
      - MODEL_PATH=/app/models/model.ckpt
    volumes:
      - ./models:/app/models

  frontend:
    build:
      context: .
      dockerfile: dockerfiles/frontend.dockerfile
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8080
    depends_on:
      - api
```

Run with:
```bash
docker-compose up
```

## CI/CD Integration

The `cloudbuild-frontend.yaml` automatically:
1. Builds Docker image on git push
2. Pushes to Artifact Registry
3. Deploys to Cloud Run

Trigger builds:

```bash
gcloud builds triggers create github \
  --name=frontend-ci \
  --repo-name=mlops-project \
  --repo-owner=your-username \
  --branch-pattern="^main$" \
  --build-config=cloudbuild-frontend.yaml
```

## Cost Optimization

- **Reduce Memory**: Start with 1GB if model is small
- **Min Instances 0**: Auto-scaling with cold starts (~30s)
- **Cloud Build Caching**: Speeds up rebuilds
- **Monitor Logs**: Identify inefficiencies

## Next Steps

- Add batch prediction endpoint
- Implement analytics dashboard
- Add model versioning and A/B testing
- Set up monitoring and alerting
- Add authentication (OAuth, API keys)

## Related Documentation

- [API Service](api.md) - FastAPI backend
- [Model Architecture](model.md) - DistilBERT implementation
- [Cloud Deployment](cloud.md) - GCP infrastructure
