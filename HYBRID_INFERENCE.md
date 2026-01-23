# Hybrid Inference Mode - API with Local Fallback

Your Streamlit frontend now supports **intelligent fallback inference**:

```
┌─────────────────────────────────────────────────┐
│        Streamlit Frontend Request                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  1. Try FastAPI        │
        │     (http://8080)      │
        └───────────┬────────────┘
                    │
         ┌──────────┴──────────┐
         │ Success?           │ No
         │                    │
         ▼                    ▼
    (Return)        ┌────────────────────┐
    Response        │ 2. Load Local Model│
                    │    (GPU/CPU)       │
                    └───────────┬────────┘
                                │
                     ┌──────────┴──────────┐
                     │ Model exists?      │ No
                     │                    │
                     ▼                    ▼
                 (Inference)      ┌─────────────────┐
                                  │ 3. Demo Mode    │
                                  │ (Keywords)      │
                                  └─────────────────┘
                                           │
                                           ▼
                                    (Demo Prediction)
```

## Three-Tier Inference System

### 1️⃣ **API Mode** (Primary)
- Calls FastAPI backend at `http://localhost:8080`
- Best for: Production, distributed systems
- Status: 🌐 "Via API"

### 2️⃣ **Local Mode** (Fallback)
- Uses model checkpoint locally: `models/model.ckpt`
- Best for: Offline operation, no API required
- Status: 🖥️ "Local Inference"

### 3️⃣ **Demo Mode** (Final Fallback)
- Keyword-based classification
- Best for: Testing without ML dependencies
- Status: 🎯 "Demo Mode"

## How It Works

**Startup:**
```python
# Load model for fallback (happens once)
local_model, tokenizer = load_local_model()
```

**Prediction:**
```python
# Try API first, then fall back intelligently
prediction, mode = predict_with_fallback(
    text,
    api_url,
    local_model,
    tokenizer
)
```

## Usage Scenarios

### Scenario 1: Both API & Local Model Available
```bash
# Terminal 1: Start API
uv run uvicorn customer_support.api:app --port 8080

# Terminal 2: Start Streamlit
uv run invoke frontend
```
→ Uses **API** (faster, offloads computation)

### Scenario 2: API Down, Local Model Available
```bash
# Terminal 1: Don't start API (or stop it)

# Terminal 2: Start Streamlit
uv run invoke frontend
```
→ Automatically falls back to **Local Model**

### Scenario 3: No API, No Model
```bash
# No services running, just Streamlit

uv run invoke frontend
```
→ Falls back to **Demo Mode** (keyword-based)

### Scenario 4: API with Model Checkpoint
```bash
export MODEL_PATH=/path/to/model.ckpt

# Terminal 1: API runs with real model
uv run uvicorn customer_support.api:app --port 8080

# Terminal 2: Streamlit with fallback capability
uv run invoke frontend
```
→ Uses **API** primarily, can fall back to **Local Model**

## Result Display

After prediction, results show:

```
Priority Level: 🔴 HIGH (85%)
Confidence: 85.0%
Inference Mode: 🌐 Via API

Model Used: DistilBERT via FastAPI Backend
```

Or if API failed:

```
Priority Level: 🔴 HIGH (88%)
Confidence: 88.0%
Inference Mode: 🖥️ Local Inference

Model Used: DistilBERT Local Inference
```

## Prediction History

History shows inference mode for each prediction:

```
Ticket 1: "System is down!"
🌐 Via API

Ticket 2: "Connection issues"
🖥️ Local Inference

Ticket 3: "How to..."
🎯 Demo Mode
```

## Benefits

✅ **Resilience**: Works even if API is down
✅ **Flexibility**: Choose best inference path automatically
✅ **Performance**: API for distributed, local for offline
✅ **Transparency**: Always know inference source
✅ **Development**: Easy testing without full stack
✅ **Production**: Graceful degradation

## Configuration

### Environment Variables

```bash
# API URL (default: http://localhost:8080)
export API_URL=http://custom-api.com:8080

# Local Model Path (default: models/model.ckpt)
export MODEL_PATH=/path/to/checkpoint.ckpt

# Start Streamlit
uv run invoke frontend
```

### Sidebar Settings

- **FastAPI URL**: Change API endpoint in real-time
- **Check API Connection**: Test API availability
- API automatically detected at startup

## Error Handling

| Scenario | Behavior |
|----------|----------|
| API unavailable | Tries local model |
| Local model missing | Uses demo mode |
| Tokenizer offline | Downloads from HuggingFace |
| All methods fail | Shows error with guidance |

## Performance Characteristics

| Mode | Latency | Requirements |
|------|---------|--------------|
| API | 100-500ms | Network + API running |
| Local | 500-2000ms | GPU/CPU + model file |
| Demo | <50ms | Keywords only |

## Advanced Usage

### Programmatic Prediction

```python
from customer_support.frontend import predict_with_fallback

text = "System is down!"
prediction, mode = predict_with_fallback(
    text,
    api_url="http://localhost:8080",
    local_model=model,
    tokenizer=tokenizer
)

print(f"Priority: {prediction['priority']}")
print(f"Inference via: {mode}")
```

### Custom Inference Flow

```python
# Try API with custom timeout
prediction = predict_via_api(text, api_url, timeout=10)

# Fall back to local
if prediction is None:
    prediction = predict_locally(text, model, tokenizer)
```

## Deployment

### Google Cloud Run

Both services can be deployed separately:

```bash
# Deploy API with fallback support
gcloud run deploy api --image=api:latest

# Deploy Frontend with intelligent fallback
gcloud run deploy frontend --image=frontend:latest \
  --set-env-vars API_URL=https://api-service.run.app
```

Frontend works even if:
- API is slow (uses local)
- API is down (uses local)
- Model not in checkpoint (uses demo)

## Future Enhancements

- 🔄 Caching layer for repeated predictions
- 📊 Analytics on inference mode usage
- 🚀 Model selection (DistilBERT vs others)
- 🧪 A/B testing between API and local
- 🌍 Geographic routing for distributed APIs

---

**Summary**: Your system now has **automatic intelligent failover** - try fast cloud API first, seamlessly fall back to local GPU/CPU inference, and always have demo mode as a safety net! 🎯
