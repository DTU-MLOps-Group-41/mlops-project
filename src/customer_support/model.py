from typing import Union
import torch
from transformers import DistilBertForSequenceClassification
from data import LABEL_MAP


def get_model(device_map: str | dict[str, Union[int, str, torch.device]] | int | torch.device = "auto",
              dtype: str | torch.dtype | None = None):
    return DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-multilingual-cased",
        num_labels=len(set(LABEL_MAP.values())),  # Ensure unique label count
        device_map=device_map,
        torch_dtype=dtype,
    )


if __name__ == "__main__":
    model = get_model()
    print(model)
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
