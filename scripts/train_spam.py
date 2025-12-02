from datetime import datetime
import json
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient

# ----- Azure ML Run Context -----
try:
    from azureml.core import Run
    run = Run.get_context()
except Exception:
    run = None

def upload_to_blob_storage(file_path: str, blob_name: str) -> str:
    """Upload file to Azure Blob Storage and return public URL"""
    try:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            print("⚠️ AZURE_STORAGE_CONNECTION_STRING not set, skipping blob upload")
            return None
        container_name = "model-outputs"
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        print(f"📤 Uploading {blob_name} to blob storage...")
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        account_name = blob_service_client.account_name
        public_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        print(f"✓ Uploaded to {public_url}")
        return public_url
    except Exception as e:
        print(f"❌ Failed to upload to blob storage: {e}")
        return None

# ----- Dataset -----
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt")
        return {"input_ids": encoding["input_ids"].flatten(), "attention_mask": encoding["attention_mask"].flatten(), "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

# ----- Model -----
class DistilBertSpamClassifier(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.distilbert.config.hidden_size, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

# ----- Training Function -----
def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    losses = []
    preds, labels = [], []
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets)
        _, predictions = torch.max(outputs, dim=1)
        losses.append(loss.item())
        preds.extend(predictions.cpu().numpy())
        labels.extend(targets.cpu().numpy())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return accuracy_score(labels, preds), sum(losses) / len(losses)

def eval_model(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets)
            _, predictions = torch.max(outputs, dim=1)
            losses.append(loss.item())
            preds.extend(predictions.cpu().numpy())
            labels.extend(targets.cpu().numpy())
    return accuracy_score(labels, preds), sum(losses) / len(losses)

# ----- Main -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-size", type=int, default=1000, help="Limit dataset size for faster training")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--dependent-var", type=str, required=True, help="Name of dependent variable column")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"🚀 DISTILBERT CUSTOM CLASSIFICATION TRAINING")
    print(f"{'='*60}")
    print(f"CSV Path: {args.csv_path}")
    print(f"Dependent Variable: {args.dependent_var}")
    print(f"Hidden Units: {args.hidden_units}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sample Size: {args.sample_size} (for faster training)")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if run:
        run.log("Device", str(device))
    
    if device.type == "cpu":
        print("⚠️  WARNING: Running on CPU. Training will be slower.")

    # ----- Load Dataset -----
    print("📦 Loading dataset...")
    df = pd.read_csv(args.csv_path)
    
    # Verify dependent variable exists
    if args.dependent_var not in df.columns:
        raise ValueError(f"Dependent variable '{args.dependent_var}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Get all columns except the dependent variable (these will be concatenated as text)
    feature_cols = [col for col in df.columns if col != args.dependent_var]
    
    # Concatenate all feature columns into a single text column
    df['text'] = df[feature_cols].astype(str).agg(' '.join, axis=1)
    
    # Convert labels to numeric if they aren't already
    label_col = df[args.dependent_var]
    unique_labels = label_col.unique()
    
    if len(unique_labels) != 2:
        raise ValueError(f"Only binary classification supported. Found {len(unique_labels)} unique labels: {unique_labels}")
    
    # Create label mapping
    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
    df['label'] = label_col.map(label_map)
    
    # Always use sample for faster training
    print(f"📊 Using sample of {args.sample_size} rows (out of {len(df)} total)")
    df = df.sample(n=min(args.sample_size, len(df)), random_state=42).reset_index(drop=True)
    
    print(f"✅ Dataset loaded: {len(df)} rows")
    print(f"   Class 0 ({unique_labels[0]}): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.1f}%)")
    print(f"   Class 1 ({unique_labels[1]}): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.1f}%)\n")

    train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = SpamDataset(train_texts.values, train_labels.values, tokenizer)
    val_dataset = SpamDataset(val_texts.values, val_labels.values, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # ----- Model -----
    print("🤖 Initializing DistilBERT model...")
    model = DistilBertSpamClassifier(args.hidden_units).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    loss_fn = nn.CrossEntropyLoss()

    print(f"📈 Training batches per epoch: {len(train_loader)}")
    print(f"📈 Total training steps: {total_steps}\n")

    if run:
        run.log("Hidden Units", args.hidden_units)
        run.log("Learning Rate", args.learning_rate)
        run.log("Epochs", args.epochs)
        run.log("Batch Size", args.batch_size)
        run.log("Total Samples", len(df))
        run.log("Dependent Variable", args.dependent_var)

    training_history = {"losses": [], "accuracies": [], "val_losses": [], "val_accuracies": [], "timestamps": []}

    # ----- Training Loop -----
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_acc, val_loss = eval_model(model, val_loader, loss_fn, device)

        training_history["losses"].append(train_loss)
        training_history["accuracies"].append(train_acc)
        training_history["val_losses"].append(val_loss)
        training_history["val_accuracies"].append(val_acc)
        training_history["timestamps"].append(datetime.now().isoformat())

        print(f"✓ Epoch {epoch + 1}/{args.epochs} Complete")
        print(f"  Train Accuracy: {train_acc:.2%}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2%}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"{'='*60}\n")

        if run:
            run.log("Epoch Loss", train_loss)
            run.log("Epoch Accuracy", train_acc)
            run.log("Val Loss", val_loss)
            run.log("Val Accuracy", val_acc)
            run.log("Epoch", epoch + 1)

    print(f"\n{'='*60}")
    print(f"🎉 Training Complete!")
    print(f"{'='*60}")
    print(f"Final Train Loss: {training_history['losses'][-1]:.4f}")
    print(f"Final Train Accuracy: {training_history['accuracies'][-1]:.2%}")
    print(f"Final Val Loss: {training_history['val_losses'][-1]:.4f}")
    print(f"Final Val Accuracy: {training_history['val_accuracies'][-1]:.2%}")
    print(f"{'='*60}\n")

    # ----- Prepare Results -----
    results = {"dataset": os.path.basename(args.csv_path), "model_type": "DistilBERT", "dependent_variable": args.dependent_var, "label_mapping": {str(k): int(v) for k, v in label_map.items()}, "samples": len(df), "class_0_count": int((df['label'] == 0).sum()), "class_1_count": int((df['label'] == 1).sum()), "hyperparameters": {"hidden_units": args.hidden_units, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "epochs": args.epochs, "sample_size": args.sample_size}, "training_history": training_history, "final_train_loss": training_history["losses"][-1], "final_train_accuracy": training_history["accuracies"][-1], "final_val_loss": training_history["val_losses"][-1], "final_val_accuracy": training_history["val_accuracies"][-1], "timestamp": datetime.now().isoformat()}

    # ----- Save Outputs -----
    print("\n📁 Saving outputs...")
    os.makedirs("outputs", exist_ok=True)

    results_path = "outputs/training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Results saved to {results_path}")

    model_path = "outputs/distilbert_spam_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "hidden_units": args.hidden_units, "label_mapping": label_map, "final_val_accuracy": results["final_val_accuracy"], "final_val_loss": results["final_val_loss"]}, model_path)
    print(f"✓ Model saved to {model_path}")

    metrics_path = "outputs/metrics.json"
    metrics_data = {"final_train_loss": float(results["final_train_loss"]), "final_train_accuracy": float(results["final_train_accuracy"]), "final_val_loss": float(results["final_val_loss"]), "final_val_accuracy": float(results["final_val_accuracy"]), "training_history": {"train_losses": [float(x) for x in training_history["losses"]], "train_accuracies": [float(x) for x in training_history["accuracies"]], "val_losses": [float(x) for x in training_history["val_losses"]], "val_accuracies": [float(x) for x in training_history["val_accuracies"]]}}
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")

    # ----- Upload to Blob Storage -----
    print("\n☁️ Uploading to Azure Blob Storage...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.getenv("AZUREML_RUN_ID", "local")
    
    model_blob_name = f"{job_id}/distilbert_custom_model_{timestamp}.pt"
    results_blob_name = f"{job_id}/training_results_{timestamp}.json"
    metrics_blob_name = f"{job_id}/metrics_{timestamp}.json"
    
    model_url = upload_to_blob_storage(model_path, model_blob_name)
    results_url = upload_to_blob_storage(results_path, results_blob_name)
    metrics_url = upload_to_blob_storage(metrics_path, metrics_blob_name)
    
    manifest = {"job_id": job_id, "timestamp": timestamp, "model_url": model_url, "results_url": results_url, "metrics_url": metrics_url, "final_train_loss": float(results["final_train_loss"]), "final_train_accuracy": float(results["final_train_accuracy"]), "final_val_loss": float(results["final_val_loss"]), "final_val_accuracy": float(results["final_val_accuracy"]), "training_history": {"train_losses": [float(x) for x in training_history["losses"]], "train_accuracies": [float(x) for x in training_history["accuracies"]], "val_losses": [float(x) for x in training_history["val_losses"]], "val_accuracies": [float(x) for x in training_history["val_accuracies"]]}}
    
    manifest_path = "outputs/manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Manifest saved to {manifest_path}")
    
    manifest_blob_name = f"{job_id}/manifest.json"
    manifest_url = upload_to_blob_storage(manifest_path, manifest_blob_name)
    
    if run:
        if model_url:
            run.log("model_url", model_url)
            print(f"✓ Model URL: {model_url}")
        if results_url:
            run.log("results_url", results_url)
            print(f"✓ Results URL: {results_url}")
        if metrics_url:
            run.log("metrics_url", metrics_url)
            print(f"✓ Metrics URL: {metrics_url}")
        if manifest_url:
            run.log("manifest_url", manifest_url)
            print(f"✓ Manifest URL: {manifest_url}")

        run.log("final_train_loss", float(results["final_train_loss"]))
        run.log("final_train_accuracy", float(results["final_train_accuracy"]))
        run.log("final_val_loss", float(results["final_val_loss"]))
        run.log("final_val_accuracy", float(results["final_val_accuracy"]))

        for i, (t_loss, t_acc, v_loss, v_acc) in enumerate(zip(training_history["losses"], training_history["accuracies"], training_history["val_losses"], training_history["val_accuracies"])):
            run.log_row("training_progress", epoch=i+1, train_loss=float(t_loss), train_accuracy=float(t_acc), val_loss=float(v_loss), val_accuracy=float(v_acc))

        print("✓ Files uploaded to Azure ML run outputs")
        
if __name__ == "__main__":
    main()