
from transformers import DistilBertForSequenceClassification
from data import LABEL_MAP

def get_model():
    return DistilBertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased",
                                                              num_labels=len(LABEL_MAP))

if __name__ == "__main__":
    model = get_model()
    print(model)
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

