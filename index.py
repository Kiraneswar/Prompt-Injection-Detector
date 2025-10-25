from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

# 1. Load and preprocess dataset
print("Loading dataset...")
dataset = load_dataset("csv", data_files="dataset.csv")

# Convert is_malicious to int type and rename to label
def preprocess_dataset(example):
    example["label"] = int(example["is_malicious"])
    return example

print("Preprocessing dataset...")
dataset = dataset.map(preprocess_dataset)

# 2. Train/Validation Split
dataset = dataset["train"].train_test_split(test_size=0.2)

# 3. Tokenizer
print("Tokenizing texts...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
dataset = dataset.rename_column("label", "labels")

# 4. Load model (DistilBERT for fine-tuning)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# 6. Define compute_metrics function
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 7. Trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 8. Train
print("Starting training...")
trainer.train()

# 9. Evaluate
print("\nEvaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

# 10. Save model
print("Saving model...")
trainer.save_model("./prompt_injection_detector")
print("Model saved successfully!")
