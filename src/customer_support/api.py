"""FastAPI application for customer support ticket priority classification."""

import glob
import os
from functools import lru_cache
from pathlib import Path
import shutil
import sys
from loguru import logger

import torch
import wandb
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer

from customer_support.model import TicketClassificationModule

# Reverse mapping from class ID to priority name
PRIORITY_NAMES = {0: "low", 1: "medium", 2: "high"}
model = None

# GCS-mounted cache directory for model storage
MODEL_CACHE_DIR = Path("/mnt/models")
CACHE_DIGEST_FILE = MODEL_CACHE_DIR / ".digest"

logger.remove()  # Remove the default logger
logger.add(sys.stdout, level="WARNING")  # Add a new logger with WARNING level


def _get_model() -> TicketClassificationModule:
    """Load model from cache if valid, otherwise download from W&B and cache."""
    api = wandb.Api()  # type: ignore
    artifact = api.artifact(os.getenv("WANDB_ARTIFACT_PATH"), type="model")
    current_digest = artifact.digest

    if not _is_cache_valid(current_digest):
        # 1. Use a local temporary directory for the download
        # Cloud Run /tmp is writable and supports chmod/renames
        temp_download_dir = Path("/tmp/model_download")
        if temp_download_dir.exists():
            shutil.rmtree(temp_download_dir)
        temp_download_dir.mkdir(parents=True)

        logger.debug(f"Downloading artifact to temporary storage: {temp_download_dir}")
        artifact.download(root=str(temp_download_dir))

        # 2. Copy files from /tmp to the GCS mount (/mnt/models)
        # We use shutil.copy2 which is more robust for GCS FUSE
        logger.debug(f"Moving model to GCS cache: {MODEL_CACHE_DIR}")
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        for item in temp_download_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, MODEL_CACHE_DIR / item.name)

        # 3. Write the digest to verify the cache later
        CACHE_DIGEST_FILE.write_text(current_digest)

        # Cleanup /tmp to save memory
        shutil.rmtree(temp_download_dir)

    # Find checkpoint file in cache directory
    ckpt_files = glob.glob(f"{MODEL_CACHE_DIR}/*.ckpt")
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt file found in {MODEL_CACHE_DIR}")
    checkpoint_path = Path(ckpt_files[0])

    loaded_model = TicketClassificationModule.load_from_checkpoint(checkpoint_path)
    loaded_model.eval()
    loaded_model.freeze()
    return loaded_model


def _is_cache_valid(current_digest: str) -> bool:
    """Check if cached model exists and matches the current W&B artifact digest."""
    if not MODEL_CACHE_DIR.exists() or not CACHE_DIGEST_FILE.exists():
        return False
    ckpt_files = glob.glob(f"{MODEL_CACHE_DIR}/*.ckpt")
    if not ckpt_files:
        return False
    cached_digest = CACHE_DIGEST_FILE.read_text().strip()
    return cached_digest == current_digest


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
