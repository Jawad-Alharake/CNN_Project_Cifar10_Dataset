import ctypes # Built-in Windows system tool

import matplotlib
matplotlib.use('Agg') # Prevents Qt interface crashes during training plots

import torch
import torch.nn as nn
import torch.optim as optim
import time # For measuring absolute elapsed runtime
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from dataset import get_data_loaders, visualize_samples
from models import CustomCNN, get_transfer_learning_model

# Early Stopping Mechanism
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Enhanced Main Engine Training & Validation Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=25):
    early_stopping = EarlyStopping(patience=5)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Start tracking total model training time
    model_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time() # Track single epoch time
        model.train()
        
        # If we are doing Transfer Learning, force frozen Batch Normalization 
        # layers to stay in .eval() mode so they don't corrupt their weights!
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
                module.eval()
        
        running_loss, correct, total = 0.0, 0, 0
        
        # Enhanced progress bar for training batches
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # Validation evaluation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            # Addition of a progress bar for the validation phase
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        # Compute how long this specific epoch took
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Results - Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        print(f"Epoch completed in {epoch_duration:.1f} seconds.\n")
        
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Halting execution loop.")
            break
            
    total_model_time = time.time() - model_start_time
    print(f"Complete training phase finished in {total_model_time/60:.2f} minutes.")
    return history



def evaluate_and_plot(model, test_loader, device, model_name="Model"):
    model.eval()
    all_preds = []
    all_labels = []
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    with torch.no_grad():
        # Addition of a progress bar for the final test set evaluation
        for inputs, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n================ Metrics for {model_name} ================")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Ground Truth Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()



def plot_history(history, model_name="Model"):
    """Generates and saves Training vs Validation Loss and Accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Left Plot: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title(f'{model_name} - Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Right Plot: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Acc')
    plt.title(f'{model_name} - Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    filename = f"{model_name.lower().replace(' ', '_')}_history.png"
    plt.savefig(filename)
    print(f"Training history plot saved as '{filename}'!")
    plt.close()



if __name__ == '__main__':
    # Windows API Constants to prevent system sleep
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002 # Optional: Keeps screen on too

    print("Activating Windows Caffeine Mode: Keeping laptop awake for training...")
    # Tell Windows: "Hey, this thread is doing heavy work. Do NOT go to sleep."
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    
    
    # Start absolute script clock
    script_start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using runtime device platform: {device}")
    
    # ---- 1. RUN CUSTOM CNN EXPERIMENT ----
    print("\n--- Starting Custom CNN Phase ---")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64, resize_for_tl=False)
    
    custom_model = CustomCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(custom_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Note: If running on a slow CPU, we can temporarily change epochs to 2 for quick testing!
    custom_history = train_model(custom_model, train_loader, val_loader, criterion, optimizer, device, epochs=30)
    plot_history(custom_history, model_name="Custom CNN from Scratch")
    evaluate_and_plot(custom_model, test_loader, device, model_name="Custom CNN from Scratch")
    
    # ---- 2. RUN TRANSFER LEARNING EXPERIMENT (MobileNetV2) ----
    print("\n--- Starting Transfer Learning Phase ---")
    tl_train_loader, tl_val_loader, tl_test_loader = get_data_loaders(batch_size=64, resize_for_tl=True)
    
    tl_model = get_transfer_learning_model(num_classes=10, feature_extract=True).to(device)
    optimizer_tl = optim.Adam(tl_model.classifier.parameters(), lr=0.0001) # The learning rate has to be smaller for transfer learning to avoid destroying pre-trained weights.
    
    tl_history = train_model(tl_model, tl_train_loader, tl_val_loader, criterion, optimizer_tl, device, epochs=20)
    plot_history(tl_history, model_name="Transfer Learning MobileNetV2")
    evaluate_and_plot(tl_model, tl_test_loader, device, model_name="Transfer Learning MobileNetV2")
    
    # Final total execution clock metric
    total_script_run_time = time.time() - script_start_time
    print("\n========================================================")
    print(f"MASTER PIPELINE COMPLETE! Total Execution Time: {total_script_run_time/60:.2f} minutes.")
    print("========================================================")
    
    print("Releasing Windows power control. The laptop can now sleep safely.")
    # Reset back to normal Windows behavior so your battery doesn't drain later
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)