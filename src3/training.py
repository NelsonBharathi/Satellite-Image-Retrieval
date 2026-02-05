"""
training.py
Intent Classifier Training Module
Fine-tunes DistilBERT for intent classification
"""

import pandas as pd
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
import torch
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

def train_intent_classifier():
    """Fine-tunes a DistilBERT model for intent classification."""
    print("=" * 70)
    print("TRAINING INTENT CLASSIFIER")
    print("=" * 70)
    
    # Load training data
    try:
        df = pd.read_csv("data/processed/synth_queries.csv")
    except FileNotFoundError:
        print("❌ Error: synth_queries.csv not found!")
        print("Please run data_preparation.py first")
        return
    
    print(f"✅ Loaded {len(df)} training queries")
    print(f"\n✅ Intent distribution:")
    print(df['intent'].value_counts())
    
    # Encode labels
    le = LabelEncoder()
    df['intent_encoded'] = le.fit_transform(df['intent'])
    print(f"\n✅ Label classes: {le.classes_}")
    
    # Initialize tokenizer and model
    print("\n✅ Loading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(le.classes_)
    )
    
    # Prepare datasets
    train_texts = df['query'].tolist()
    train_labels = df['intent_encoded'].tolist()
    
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    train_dataset = IntentDataset(train_encodings, train_labels)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./models/intent_results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./models/logs',
        logging_steps=10,
        save_strategy='epoch',
        eval_strategy='no'
    )
    
    # Train
    print("\n✅ Starting training...")
    print("This may take 5-15 minutes depending on your hardware...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()
    
    # Save model and tokenizer
    model_path = Path('./models/intent_classifier')
    model_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    joblib.dump(le, './models/label_encoder.joblib')
    
    print("\n" + "=" * 70)
    print("INTENT CLASSIFIER TRAINING COMPLETE")
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Label encoder saved to: models/label_encoder.joblib")
    print("=" * 70)

def run_training():
    """Runs all training steps."""
    Path("models").mkdir(exist_ok=True)
    train_intent_classifier()
    print("\n✅ All training complete!")

if __name__ == "__main__":
    run_training()