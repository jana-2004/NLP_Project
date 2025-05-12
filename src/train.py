"""
Training pipeline for mental health classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
from model import MentalHealthClassifier
from data_processing import load_and_process_data

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128


class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        #  Use reset-indexed Series
        text = str(self.texts[index])
        label = int(self.labels[index])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, digits=4)
    return report


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading and processing data...")
    train_df, test_df, label_encoder = load_and_process_data("data/raw/mental_health.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = MentalHealthDataset(train_df["statement"], train_df["label"], tokenizer, MAX_LEN)
    test_dataset = MentalHealthDataset(test_df["statement"], test_df["label"], tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MentalHealthClassifier(MODEL_NAME, num_labels=len(label_encoder.classes_))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS)

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")

    print("Evaluating on test set...")
    report = eval_model(model, test_loader, device)
    print(report)

    print("Saving model...")
    torch.save(model.state_dict(), "models/model_v1.pt")
    print("Done.")
