from datetime import datetime
import json
import os
import argparse
import zipfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
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

# ----- Custom Dataset -----
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with class folders
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Build file list
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    self.samples.append((str(img_path), class_idx))
        
        print(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), label
            return Image.new('RGB', (224, 224)), label

# ----- Training Functions -----
def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    losses = []
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        losses.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    accuracy = correct / total
    avg_loss = sum(losses) / len(losses)
    return accuracy, avg_loss

def eval_model(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            losses.append(loss.item())
    
    accuracy = correct / total
    avg_loss = sum(losses) / len(losses)
    return accuracy, avg_loss

# ----- Main -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="mobilenet", choices=["mobilenet", "shufflenet"], help="Type of CNN model to use")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset ZIP file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if run:
        run.log("Device", str(device))
    

    # ----- Extract Dataset -----
    extract_dir = "dataset"
    
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(args.dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✓ Dataset extracted to {extract_dir}")
    except Exception as e:
        print(f"❌ Failed to extract dataset: {e}")
        raise

    # Find the actual data directory (handle nested folders)
    data_dirs = [d for d in Path(extract_dir).rglob('*') if d.is_dir() and any(d.iterdir())]
    
    # Find directory with class folders
    dataset_root = None
    for d in data_dirs:
        subdirs = [x for x in d.iterdir() if x.is_dir()]
        if len(subdirs) >= 2:  # At least 2 class folders
            # Check if subdirs contain images
            has_images = False
            for subdir in subdirs:
                image_files = list(subdir.glob('*.jpg')) + list(subdir.glob('*.jpeg')) + list(subdir.glob('*.png'))
                if image_files:
                    has_images = True
                    break
            if has_images:
                dataset_root = d
                break
    
    if dataset_root is None:
        # Try the extract_dir itself
        subdirs = [x for x in Path(extract_dir).iterdir() if x.is_dir()]
        if len(subdirs) >= 2:
            dataset_root = Path(extract_dir)
        else:
            raise ValueError(f"Could not find valid dataset structure in {extract_dir}")
    
    print(f"Using dataset root: {dataset_root}")

    # ----- Data Transforms -----
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ----- Load Dataset -----
    print("📂 Loading custom dataset...")
    full_dataset = CustomImageDataset(dataset_root, transform=transform_train)
    
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    total_images = len(full_dataset)
    
    print(f"✅ Dataset loaded:")
    print(f"   Total images: {total_images}")
    print(f"   Classes: {num_classes}")
    print(f"   Class names: {class_names}\n")

    # Split dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = transform_val

    print(f"📊 Data split:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}\n")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ----- Model -----
    print("🤖 Initializing model...")
    if args.model_type == "mobilenet":
        model = models.mobilenet_v3_small(weights=None, num_classes=num_classes)
        print("📱 Using MobileNetV3-Small")
    else:
        model = models.shufflenet_v2_x1_0(weights=None, num_classes=num_classes)
        print("🔀 Using ShuffleNetV2")
    
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    print(f"📈 Training batches per epoch: {len(train_loader)}")
    print(f"📈 Validation batches per epoch: {len(val_loader)}\n")

    if run:
        run.log("Model Type", args.model_type)
        run.log("Learning Rate", args.lr)
        run.log("Epochs", args.epochs)
        run.log("Batch Size", args.batch_size)
        run.log("Total Samples", total_images)
        run.log("Num Classes", num_classes)
        run.log("Total Parameters", total_params)

    training_history = {
        "losses": [], 
        "accuracies": [], 
        "val_losses": [], 
        "val_accuracies": [], 
        "timestamps": [],
        "learning_rates": []
    }

    # ----- Training Loop -----
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_acc, val_loss = eval_model(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']

        training_history["losses"].append(train_loss)
        training_history["accuracies"].append(train_acc)
        training_history["val_losses"].append(val_loss)
        training_history["val_accuracies"].append(val_acc)
        training_history["timestamps"].append(datetime.now().isoformat())
        training_history["learning_rates"].append(current_lr)

        print(f"✓ Epoch {epoch + 1}/{args.epochs} Complete")
        print(f"  Train Accuracy: {train_acc:.2%}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2%}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"{'='*60}\n")

        # Adjust learning rate
        scheduler.step(val_loss)

        if run:
            run.log("Epoch Loss", train_loss)
            run.log("Epoch Accuracy", train_acc)
            run.log("Val Loss", val_loss)
            run.log("Val Accuracy", val_acc)
            run.log("Learning Rate", current_lr)
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
    results = {
        "dataset": os.path.basename(args.dataset_path),
        "model_type": args.model_type,
        "num_classes": num_classes,
        "class_names": class_names,
        "total_samples": total_images,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "hyperparameters": {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
        "training_history": training_history,
        "final_train_loss": training_history["losses"][-1],
        "final_train_accuracy": training_history["accuracies"][-1],
        "final_val_loss": training_history["val_losses"][-1],
        "final_val_accuracy": training_history["val_accuracies"][-1],
        "timestamp": datetime.now().isoformat(),
    }

    # ----- Save Outputs -----
    print("\n📁 Saving outputs...")
    os.makedirs("outputs", exist_ok=True)

    results_path = "outputs/training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Results saved to {results_path}")

    model_path = "outputs/cnn_custom_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model_type,
        'num_classes': num_classes,
        'class_names': class_names,
        'final_val_accuracy': results["final_val_accuracy"],
        'final_val_loss': results["final_val_loss"],
    }, model_path)
    print(f"✓ Model saved to {model_path}")

    metrics_path = "outputs/metrics.json"
    metrics_data = {
        "final_train_loss": float(results["final_train_loss"]),
        "final_train_accuracy": float(results["final_train_accuracy"]),
        "final_val_loss": float(results["final_val_loss"]),
        "final_val_accuracy": float(results["final_val_accuracy"]),
        "training_history": {
            "train_losses": [float(x) for x in training_history["losses"]],
            "train_accuracies": [float(x) for x in training_history["accuracies"]],
            "val_losses": [float(x) for x in training_history["val_losses"]],
            "val_accuracies": [float(x) for x in training_history["val_accuracies"]]
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")

    # ----- Upload to Blob Storage -----
    print("\n☁️ Uploading to Azure Blob Storage...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.getenv("AZUREML_RUN_ID", "local")
    
    model_blob_name = f"{job_id}/cnn_custom_model_{timestamp}.pth"
    results_blob_name = f"{job_id}/training_results_{timestamp}.json"
    metrics_blob_name = f"{job_id}/metrics_{timestamp}.json"
    
    model_url = upload_to_blob_storage(model_path, model_blob_name)
    results_url = upload_to_blob_storage(results_path, results_blob_name)
    metrics_url = upload_to_blob_storage(metrics_path, metrics_blob_name)
    
    manifest = {
        "job_id": job_id,
        "timestamp": timestamp,
        "model_url": model_url,
        "results_url": results_url,
        "metrics_url": metrics_url,
        "final_train_loss": float(results["final_train_loss"]),
        "final_train_accuracy": float(results["final_train_accuracy"]),
        "final_val_loss": float(results["final_val_loss"]),
        "final_val_accuracy": float(results["final_val_accuracy"]),
        "training_history": {
            "train_losses": [float(x) for x in training_history["losses"]],
            "train_accuracies": [float(x) for x in training_history["accuracies"]],
            "val_losses": [float(x) for x in training_history["val_losses"]],
            "val_accuracies": [float(x) for x in training_history["val_accuracies"]]
        }
    }
    
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

        for i, (t_loss, t_acc, v_loss, v_acc) in enumerate(zip(
            training_history["losses"], 
            training_history["accuracies"], 
            training_history["val_losses"], 
            training_history["val_accuracies"]
        )):
            run.log_row("training_progress", 
                       epoch=i+1, 
                       train_loss=float(t_loss), 
                       train_accuracy=float(t_acc), 
                       val_loss=float(v_loss), 
                       val_accuracy=float(v_acc))

        print("✓ Files uploaded to Azure ML run outputs")
        
if __name__ == "__main__":
    main()