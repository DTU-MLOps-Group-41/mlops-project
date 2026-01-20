"""FastAPI application for customer support ticket priority classification."""

import os
from functools import lru_cache

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer

from customer_support.model import TicketClassificationModule

# Reverse mapping from class ID to priority name
PRIORITY_NAMES = {0: "low", 1: "medium", 2: "high"}

app = FastAPI(
    title="Customer Support Ticket Classifier",
    description="API for classifying customer support ticket priority using DistilBERT",
    version="1.0.0",
)


class TicketRequest(BaseModel):
    """Request model for ticket classification."""

    text: str = Field(..., min_length=1, description="The ticket body text to classify")


class TicketResponse(BaseModel):
    """Response model for ticket classification."""

    priority: str = Field(..., description="Priority level: low, medium, or high")
    priority_id: int = Field(..., description="Priority class ID: 0, 1, or 2")
    confidence: float = Field(..., description="Confidence score (probability)")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool


@lru_cache(maxsize=1)
def get_model() -> TicketClassificationModule:
    """Load and cache the model from checkpoint.

    Uses MODEL_PATH environment variable, defaulting to models/model_full.ckpt.
    """
    model_path = os.environ.get("MODEL_PATH", "models/model_full.ckpt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = TicketClassificationModule.load_from_checkpoint(model_path)
    model.eval()
    model.freeze()
    return model


@lru_cache(maxsize=1)
def get_tokenizer() -> DistilBertTokenizer:
    """Load and cache the tokenizer."""
    return DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")


@app.get("/")
def read_root():
    """Root endpoint returning API information."""
    return {
        "service": "Customer Support Ticket Classifier",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for container orchestration."""
    try:
        _ = get_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=TicketResponse)
def predict(request: TicketRequest):
    """Classify ticket priority from text.

    Args:
        request: TicketRequest with text field

    Returns:
        TicketResponse with priority, priority_id, and confidence
    """
    try:
        model = get_model()
        tokenizer = get_tokenizer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Tokenize input
    encoding = tokenizer(
        request.text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    # Run inference
    with torch.inference_mode():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

    # Get prediction and confidence
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class].item()

    return TicketResponse(
        priority=PRIORITY_NAMES[predicted_class],
        priority_id=predicted_class,
        confidence=confidence,
    )
