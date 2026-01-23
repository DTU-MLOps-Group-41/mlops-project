# API Service

The project includes a FastAPI REST service for real-time ticket priority classification.

## Overview

The API provides a simple interface for classifying customer support tickets:

- **Framework**: FastAPI with Uvicorn
- **Model**: DistilBERT loaded from checkpoint
- **Port**: 8080 (default)

## Endpoints

### `GET /`

Returns API information and available endpoints.

**Response:**
```json
{
  "service": "Customer Support Ticket Classifier",
  "version": "1.0.0",
  "endpoints": {
    "predict": "POST /predict",
    "health": "GET /health"
  }
}
```

### `GET /health`

Health check endpoint for container orchestration (Kubernetes, Cloud Run, etc.).

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`

Classify a ticket's priority from its text.

**Request Body:**
```json
{
  "text": "My computer won't start and I have an important presentation in 1 hour!"
}
```

**Response:**
```json
{
  "priority": "high",
  "priority_id": 2,
  "confidence": 0.87
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `priority` | string | Priority level: `"low"`, `"medium"`, or `"high"` |
| `priority_id` | int | Numeric class ID: 0, 1, or 2 |
| `confidence` | float | Model confidence (probability) |

**Error Responses:**

- `422 Validation Error` - Missing or empty text field
- `503 Service Unavailable` - Model checkpoint not found

## Running Locally

### Prerequisites

- Trained model checkpoint at `models/model.ckpt` (or set `MODEL_PATH` env var)

### Start the server

```bash
# Default: loads model from models/model.ckpt
uvicorn customer_support.api:app --host 0.0.0.0 --port 8080

# With custom model path
MODEL_PATH=/path/to/checkpoint.ckpt uvicorn customer_support.api:app --port 8080

# Development mode with auto-reload
uvicorn customer_support.api:app --reload --port 8080
```

### Test the API

```bash
# Health check
curl http://localhost:8080/health

# Classify a ticket
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Cannot login to my account, need help urgently"}'
```

## Docker Deployment

### Build the image

```bash
docker build -t api:latest -f dockerfiles/api.dockerfile .
```

### Run the container

```bash
docker run -p 8080:8080 \
  -v /path/to/models:/app/models \
  -e MODEL_PATH=/app/models/model.ckpt \
  api:latest
```

### With GPU support

```bash
# Build with CUDA
docker build -t api:gpu -f dockerfiles/api.dockerfile --build-arg DEVICE=cu128 .

# Run with GPU
docker run --gpus all -p 8080:8080 \
  -v /path/to/models:/app/models \
  -e MODEL_PATH=/app/models/model.ckpt \
  api:gpu
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/model.ckpt` | Path to model checkpoint |

## API Implementation Details

The API is implemented in `src/customer_support/api.py`:

- **Model loading**: Loaded once at startup via FastAPI lifespan
- **Tokenizer caching**: LRU cache for the DistilBERT tokenizer
- **Inference mode**: Uses `torch.inference_mode()` for faster predictions
- **Non-root user**: Docker container runs as unprivileged user for security

## Example: Python Client

```python
import requests

def classify_ticket(text: str, api_url: str = "http://localhost:8080") -> dict:
    response = requests.post(
        f"{api_url}/predict",
        json={"text": text}
    )
    response.raise_for_status()
    return response.json()

# Usage
result = classify_ticket("My laptop screen is broken")
print(f"Priority: {result['priority']} (confidence: {result['confidence']:.2%})")
```

## Related Documentation

- [Model](model.md) - Model architecture details
- [Cloud](cloud.md) - Deploying to GCP
