# Running Streamlit + FastAPI Together

This guide shows how to run the Streamlit frontend and FastAPI backend together for the ticket classification system.

## Quick Start (2 Terminals)

### Terminal 1: Start FastAPI Backend

```bash
uv run uvicorn customer_support.api:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at `http://localhost:8080`

**Health Check:**
```bash
curl http://localhost:8080/health
```

### Terminal 2: Start Streamlit Frontend

```bash
uv run invoke frontend
```

or directly:

```bash
uv run streamlit run src/customer_support/frontend.py
```

The frontend opens at `http://localhost:8501`

## Architecture

```
User Browser
    ↓
Streamlit UI (Port 8501)
    ↓
HTTP POST /predict
    ↓
FastAPI Backend (Port 8080)
    ↓
TicketClassificationModule
    ↓
Model Inference (GPU/CPU)
    ↓
Response (priority, confidence)
```

## Configuration

### API URL

The frontend looks for the API at:
1. **Sidebar Input** - Configure API URL in the Streamlit sidebar
2. **Environment Variable** - `API_URL` env var (default: `http://localhost:8080`)

```bash
# Set custom API URL
export API_URL=http://api.example.com:8080
uv run invoke frontend
```

### Model Checkpoint

Ensure the model checkpoint is available for the FastAPI backend:

```bash
# Check model exists
ls -lh models/model.ckpt

# Or set custom path
export MODEL_PATH=/path/to/model.ckpt
uv run uvicorn customer_support.api:app --port 8080
```

## Troubleshooting

### "Failed to connect to API"

1. **Check API is running:**
   ```bash
   curl http://localhost:8080/health
   ```

2. **Check API URL in Streamlit sidebar** - Default is `http://localhost:8080`

3. **Check model checkpoint exists:**
   ```bash
   ls models/model.ckpt
   ```

### Slow predictions

- **Increase timeout**: API has 30-second timeout, model inference may take time
- **Check CPU/GPU**: Monitor resource usage
- **Reduce model complexity**: Use smaller batch sizes if needed

### Connection timeouts

- **Check network**: Ensure `localhost:8080` is reachable
- **Check firewall**: May need to allow ports 8080 and 8501
- **Docker**: If using Docker, ensure ports are exposed

## Production Deployment

### Google Cloud Run (2 Services)

**1. Deploy FastAPI Backend:**
```bash
gcloud run deploy api \
  --image gcr.io/${PROJECT}/api:latest \
  --port 8080 \
  --memory 2Gi
```

**2. Deploy Streamlit Frontend:**
```bash
export API_URL=https://api-xxxxx.run.app

gcloud run deploy frontend \
  --image gcr.io/${PROJECT}/frontend:latest \
  --port 8501 \
  --set-env-vars API_URL=$API_URL \
  --memory 1Gi
```

### Docker Compose (Local Testing)

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

## API Endpoints

The FastAPI backend provides:

- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Classify ticket

**Example:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My server is down!"}'

# Response:
# {
#   "priority": "high",
#   "priority_id": 2,
#   "confidence": 0.92
# }
```

## Features

✅ **Streamlit Frontend**
- Clean, interactive UI
- Real-time predictions
- Prediction history
- API connection testing
- Configuration in sidebar

✅ **FastAPI Backend**
- Production-ready REST API
- Model caching
- Health checks
- Error handling
- Async support

✅ **Deployment Ready**
- Docker support
- Cloud Run compatible
- Configurable via env vars
- Graceful error handling

## Development Workflow

```bash
# 1. Make changes to frontend
vim src/customer_support/frontend.py

# 2. Streamlit auto-reloads on save

# 3. Test predictions

# 4. Make model changes
vim src/customer_support/model.py

# 5. Retrain and update checkpoint
uv run invoke train

# 6. Restart API backend
# (Ctrl+C and rerun in Terminal 1)

# 7. FastAPI auto-reloads with --reload flag
```

## Performance Tips

1. **API Response Time**: ~100-500ms (depends on GPU/CPU)
2. **Streamlit Caching**: Frontend caches API responses to avoid duplicate calls
3. **Batch Predictions**: For many predictions, consider batch API endpoint (future enhancement)
4. **Model Optimization**: Use quantization or pruning for faster inference

## Next Steps

- Add batch prediction endpoint to API
- Implement prediction analytics dashboard
- Add model versioning and A/B testing
- Set up monitoring and logging
- Add authentication to API (OAuth, API keys)

For more details:
- [API Documentation](docs/source/api.md)
- [Model Architecture](docs/source/model.md)
- [Deployment Guide](DEPLOYMENT_FRONTEND.md)
