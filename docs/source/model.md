# Model Architecture

This project uses a fine-tuned DistilBERT model for customer support ticket priority classification.

## Overview

The model classifies customer support tickets into three priority levels:

| Class ID | Priority | Description |
|----------|----------|-------------|
| 0 | Low | Non-urgent tickets |
| 1 | Medium | Standard priority |
| 2 | High | Urgent tickets requiring immediate attention |

## Why DistilBERT?

[DistilBERT](https://huggingface.co/distilbert-base-multilingual-cased) was chosen for several reasons:

- **Multilingual support**: Handles tickets in multiple languages (the dataset is multilingual)
- **Efficiency**: 40% smaller and 60% faster than BERT while retaining 97% of its performance
- **Pre-trained**: Leverages transfer learning from massive text corpora
- **Production-ready**: Suitable for real-time inference in production environments

Model: `distilbert-base-multilingual-cased`
- Parameters: ~66M (distilled from BERT's 110M)
- Vocabulary: 119,547 tokens
- Max sequence length: 512 tokens

## TicketClassificationModule

The model is implemented as a PyTorch Lightning module in `src/customer_support/model.py`.

### Architecture

```
Input Text
    ↓
DistilBERT Tokenizer (512 max tokens)
    ↓
DistilBERT Encoder (6 transformer layers)
    ↓
Classification Head (Linear: 768 → 3)
    ↓
Softmax
    ↓
Priority Prediction (0, 1, or 2)
```

### Key Features

- **PyTorch Lightning integration**: Automatic training/validation/test loops
- **Mixed precision training**: 16-bit floating point for faster training
- **Distributed training support**: Scales across multiple GPUs
- **TorchMetrics**: Accurate metric computation with distributed sync
- **Configurable LR scheduling**: Optional warmup and decay

### Constructor Parameters

```python
TicketClassificationModule(
    model_name="distilbert-base-multilingual-cased",  # HuggingFace model ID
    num_classes=3,                                     # Output classes
    learning_rate=5e-5,                                # AdamW learning rate
    weight_decay=0.01,                                 # L2 regularization
    lr_scheduler_config=None,                          # Optional LR scheduler
)
```

### Usage

```python
from customer_support.model import TicketClassificationModule

# Initialize model
model = TicketClassificationModule(learning_rate=5e-5)

# Load from checkpoint
model = TicketClassificationModule.load_from_checkpoint("path/to/checkpoint.ckpt")

# Inference
model.eval()
with torch.inference_mode():
    outputs = model(input_ids=tokens, attention_mask=mask)
    predictions = torch.argmax(outputs.logits, dim=-1)
```

## Input/Output Format

### Input

The model expects tokenized input from the DistilBERT tokenizer:

| Field | Shape | Description |
|-------|-------|-------------|
| `input_ids` | `[batch, seq_len]` | Token IDs |
| `attention_mask` | `[batch, seq_len]` | Attention mask (1=real, 0=padding) |
| `labels` | `[batch]` | Ground truth labels (training only) |

### Output

The model returns a `SequenceClassifierOutput` with:

| Field | Shape | Description |
|-------|-------|-------------|
| `loss` | `[]` | Cross-entropy loss (if labels provided) |
| `logits` | `[batch, 3]` | Raw logits for each class |

## Metrics

Training logs the following metrics to W&B and CSV:

| Metric | Description |
|--------|-------------|
| `train_loss` | Training loss (per step and epoch) |
| `train_accuracy` | Training accuracy (per epoch) |
| `val_loss` | Validation loss (per epoch) |
| `val_accuracy` | Validation accuracy (per epoch) |
| `test_loss` | Test loss (final evaluation) |
| `test_accuracy` | Test accuracy (final evaluation) |

## Checkpointing

Model checkpoints are saved automatically during training:

- **Location**: `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/`
- **Format**: `epoch={epoch}-step={step}.ckpt`
- **Selection**: Best model by `val_accuracy` (configurable)
- **Contents**: Model weights, optimizer state, hyperparameters

### Loading Checkpoints

```python
# For inference
model = TicketClassificationModule.load_from_checkpoint(
    "checkpoints/epoch=5-step=1000.ckpt",
    local_files_only=True,  # Don't download model weights
)

# Resume training
trainer.fit(model, datamodule, ckpt_path="path/to/checkpoint.ckpt")
```

## Optimizer

The model uses AdamW optimizer with:

- **Learning rate**: Configurable (default: 5e-5)
- **Weight decay**: 0.01 (L2 regularization)
- **Optional LR scheduler**: Cosine decay with warmup (production config)

## Related Documentation

- [Training](training.md) - How to train the model
- [API](api.md) - Using the model for inference
- [Cloud](cloud.md) - Training on Vertex AI
