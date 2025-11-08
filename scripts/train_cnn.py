import os
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image
import torchvision.models as models

from azureml.core.run import Run
from azure.storage.blob import BlobServiceClient

run = Run.get_context()

def upload_to_blob_storage(file_path: str, blob_name: str) -> str:
    try:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            print("⚠️ AZURE_STORAGE_CONNECTION_STRING not set, skipping blob upload")
            return None
        container_name = "model-outputs"
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        try:
            container_client.get_container_properties()
        except:
            container_client.create_container(public_access='blob')
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        account_name = blob_service_client.account_name
        public_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        print(f"✓ Uploaded to {public_url}")
        return public_url
    except Exception as e:
        print(f"❌ Failed to upload to blob storage: {e}")
        return None

class CNNTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        run.log("Device", str(self.device))
    
    def train_model(self, 
                    train_loader: DataLoader,
                    model_type: str = "mobilenet",
                    learning_rate: float = 0.001,
                    epochs: int = 2,
                    num_classes: int = 10) -> Dict:

        print(f"\n{'='*60}")
        print(f"Starting Fast CNN Training on Azure ML")
        print(f"{'='*60}")
        print(f"Model Type: {model_type}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Epochs: {epochs}")
        print(f"Training Samples: {len(train_loader.dataset)}")
        print(f"Batch Size: {train_loader.batch_size}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        run.log("Model Type", model_type)
        run.log("Learning Rate", learning_rate)
        run.log("Epochs", epochs)
        run.log("Total Samples", len(train_loader.dataset))
        run.log("Batch Size", train_loader.batch_size)

        # Create model - using efficient architectures
        if model_type == "simple":
            model = models.mobilenet_v3_small(weights=None, num_classes=num_classes)
            print("📱 Using MobileNetV3-Small (optimized for speed)")
        else:  # traffic
            model = models.shufflenet_v2_x1_0(weights=None, num_classes=num_classes)
            print("🔀 Using ShuffleNetV2 (very fast inference)")
        
        model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        run.log("Total Parameters", total_params)
        run.log("Trainable Parameters", trainable_params)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler for better convergence
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        training_history = {
            "losses": [],
            "accuracies": [],
            "timestamps": [],
            "learning_rates": []
        }

        total_steps = len(train_loader) * epochs
        current_step = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                epoch_loss += loss.item()
                current_step += 1
                progress = (current_step / total_steps) * 100

                # Log every 10% of batches
                if (batch_idx + 1) % max(1, len(train_loader)//10) == 0:
                    batch_acc = (predicted == labels).sum().item() / labels.size(0)
                    print(f"Epoch [{epoch+1}/{epochs}] "
                          f"Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f} | "
                          f"Accuracy: {batch_acc:.2%} | "
                          f"Progress: {progress:.1f}%")

                    run.log("Batch Loss", loss.item())
                    run.log("Batch Accuracy", batch_acc)
                    run.log("Progress", progress)

            # Epoch statistics
            avg_loss = epoch_loss / len(train_loader)
            epoch_accuracy = correct / total
            current_lr = optimizer.param_groups[0]['lr']

            training_history["losses"].append(avg_loss)
            training_history["accuracies"].append(epoch_accuracy)
            training_history["timestamps"].append(datetime.now().isoformat())
            training_history["learning_rates"].append(current_lr)

            print(f"\n{'='*60}")
            print(f"✓ Epoch {epoch+1}/{epochs} Complete")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {epoch_accuracy:.2%}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"{'='*60}\n")

            run.log("Epoch Loss", avg_loss)
            run.log("Epoch Accuracy", epoch_accuracy)
            run.log("Epoch", epoch + 1)
            run.log("Learning Rate", current_lr)
            
            # Adjust learning rate
            scheduler.step(avg_loss)

        print(f"\n{'='*60}")
        print(f"🎉 Training Complete!")
        print(f"{'='*60}")
        print(f"Final Loss: {training_history['losses'][-1]:.4f}")
        print(f"Final Accuracy: {training_history['accuracies'][-1]:.2%}")
        print(f"{'='*60}\n")

        run.log("Final Loss", training_history['losses'][-1])
        run.log("Final Accuracy", training_history['accuracies'][-1])

        return training_history, model

def main():
    parser = argparse.ArgumentParser(description="Train Fast CNN on Azure ML")
    parser.add_argument("--model-type", type=str, default="simple", 
                        choices=["simple", "traffic"],
                        help="Model architecture: 'simple' (MobileNetV3-Small, fastest) or 'traffic' (ShuffleNetV2, very fast)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--num-samples", type=int, default=1000, 
                        help="Number of training samples (max 50000 for CIFAR-10)")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"],
                        help="Dataset to use")
    
    args = parser.parse_args()

    print("\n" + "="*60)
    print("🚀 FAST CNN TRAINING PIPELINE")
    print("="*60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: {args.model_type}")
    print(f"Samples: {args.num_samples}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("="*60 + "\n")

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load dataset
    print("📦 Loading dataset...")
    if args.dataset == "cifar10":
        full_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform_train
        )
        num_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        full_dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transform_train
        )
        num_classes = 100
        class_names = full_dataset.classes

    # Limit dataset size
    total_available = len(full_dataset)
    num_samples = min(args.num_samples, total_available)
    
    # Create subset
    indices = torch.randperm(total_available)[:num_samples].tolist()
    dataset = Subset(full_dataset, indices)
    
    print(f"✅ Dataset loaded: {num_samples} samples from {args.dataset.upper()}")
    print(f"   Total available: {total_available}")
    print(f"   Classes: {num_classes}")
    print(f"   Image size: 32x32x3\n")

    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    print(f"📊 Created DataLoader:")
    print(f"   Batches per epoch: {len(train_loader)}")
    print(f"   Last batch size: {num_samples % args.batch_size or args.batch_size}\n")

    # Initialize trainer
    print("🤖 Initializing trainer...")
    trainer = CNNTrainer()

    # Train model
    history, model = trainer.train_model(
        train_loader=train_loader,
        model_type=args.model_type,
        learning_rate=args.lr,
        epochs=args.epochs,
        num_classes=num_classes
    )

    # Prepare results
    results = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "samples": num_samples,
        "total_available": total_available,
        "num_classes": num_classes,
        "class_names": class_names,
        "hyperparameters": {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
        "training_history": history,
        "final_loss": history["losses"][-1],
        "final_accuracy": history["accuracies"][-1],
        "timestamp": datetime.now().isoformat(),
    }

    # Save outputs
    print("\n📁 Saving outputs...")
    os.makedirs("outputs", exist_ok=True)

    # Save results
    results_path = "outputs/training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Results saved to {results_path}")

    # Save model
    model_path = "outputs/cnn_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model_type,
        'num_classes': num_classes,
        'class_names': class_names,
        'final_accuracy': results["final_accuracy"],
        'final_loss': results["final_loss"],
    }, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # CRITICAL: Write metrics to a separate file for easy parsing
    metrics_path = "outputs/metrics.json"
    metrics_data = {
        "final_loss": float(results["final_loss"]),
        "final_accuracy": float(results["final_accuracy"]),
        "training_history": {
            "losses": [float(x) for x in history["losses"]],
            "accuracies": [float(x) for x in history["accuracies"]]
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")

    # Upload to Blob Storage for public access
    print("\n☁️ Uploading to Azure Blob Storage...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.getenv("AZUREML_RUN_ID", "local")
    
    model_blob_name = f"{job_id}/cnn_model_{timestamp}.pth"
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
        "final_loss": float(results["final_loss"]),
        "final_accuracy": float(results["final_accuracy"]),
        "training_history": {
            "losses": [float(x) for x in history["losses"]],
            "accuracies": [float(x) for x in history["accuracies"]]
        }
    }
    
    manifest_path = "outputs/manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Manifest saved to {manifest_path}")
    
    manifest_blob_name = f"{job_id}/manifest.json"
    manifest_url = upload_to_blob_storage(manifest_path, manifest_blob_name)
    
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

    run.log("final_loss", float(results["final_loss"]))
    run.log("final_accuracy", float(results["final_accuracy"]))

    for i, (loss, acc) in enumerate(zip(history["losses"], history["accuracies"])):
        run.log_row("training_progress", epoch=i+1, loss=float(loss), accuracy=float(acc))

    run.upload_file(name="outputs/training_results.json", path_or_stream=results_path)
    run.upload_file(name="outputs/cnn_model.pth", path_or_stream=model_path)
    run.upload_file(name="outputs/metrics.json", path_or_stream=metrics_path)
    run.upload_file(name="outputs/manifest.json", path_or_stream=manifest_path)

    print("✓ Files uploaded to Azure ML run outputs")
    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print(f"Dataset: {args.dataset.upper()} ({num_samples} samples)")
    print(f"Model: {args.model_type}")
    print(f"Final Loss: {results['final_loss']:.4f}")
    print(f"Final Accuracy: {results['final_accuracy']:.2%}")
    print(f"\nPublic URLs:")
    if model_url:
        print(f"  Model: {model_url}")
    if results_url:
        print(f"  Results: {results_url}")
    if manifest_url:
        print(f"  Manifest: {manifest_url}")
    print("="*60 + "\n")
    print("✓ All done! Model and results available publicly via blob storage!")

if __name__ == "__main__":
    main()