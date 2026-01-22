"""FastAPI application for customer support ticket priority classification."""

import glob
import os
from functools import lru_cache
from fastapi.concurrency import asynccontextmanager
import wandb

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer

from customer_support.model import TicketClassificationModule

# Reverse mapping from class ID to priority name
PRIORITY_NAMES = {0: "low", 1: "medium", 2: "high"}
model = None


def _get_model() -> TicketClassificationModule:
    """Load and cache the model from checkpoint."""
    run = wandb.Api()  # type: ignore[attr-defined]
    artifact = run.artifact(os.getenv("WANDB_ARTIFACT_PATH"), type="model")
    artifact_dir = artifact.download()
    # Find the .ckpt file in the artifact directory
    ckpt_files = glob.glob(f"{artifact_dir}/*.ckpt")
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt file found in {artifact_dir}")
    checkpoint_path = ckpt_files[0]
    model = TicketClassificationModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = _get_model()
    yield


app = FastAPI(
    title="Customer Support Ticket Classifier",
    description="API for classifying customer support ticket priority using DistilBERT",
    version="1.0.0",
    lifespan=lifespan,
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
    model_loaded = model is not None
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
        # model = _get_model()
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
        assert model is not None
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
