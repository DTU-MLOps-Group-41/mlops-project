from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from customer_support.data import TicketDataset
from customer_support.model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train():
    """Train the customer support ticket classifier.

    This is a placeholder implementation showing how to use TicketDataset with DataLoader.
    """

    train_dataset = TicketDataset(root="data", split="train", dataset_type="small")
    val_dataset = TicketDataset(root="data", split="validation", dataset_type="small")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = get_model().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Model: {model}")

    for epoch in range(3):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            # attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}")

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()


if __name__ == "__main__":
    train()
