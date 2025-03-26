import torch
import torch.nn as nn
from tqdm.notebook import tqdm # Use notebook version if running in Jupyter
# from tqdm import tqdm # Use standard version if running as script
from sklearn.metrics import roc_auc_score
import numpy as np

# Import utility functions
from .utils import save_model, save_training_log, plot_training_history

def train_epoch(model, dataloader, optimizer, criterion, device, config):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        try:
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            if 'label' not in inputs:
                 print(f"Warning: 'label' not found in batch keys: {batch.keys()}")
                 continue # Skip batch if no label
            labels = inputs.pop('label')

            optimizer.zero_grad()
            outputs = model(**inputs) # Unpack filtered inputs
            loss = criterion(outputs, labels)

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_acc = (predicted == labels).sum().item() / labels.size(0) if labels.size(0) > 0 else 0
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{batch_acc:.4f}"})

        except Exception as e:
            print(f"\nError during training batch: {e}")
            print(f"Batch keys: {batch.keys()}")
            # Optionally skip batch or raise error
            continue

    avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def evaluate_epoch(model, dataloader, criterion, device, config):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            try:
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                if 'label' not in inputs:
                     print(f"Warning: 'label' not found in eval batch keys: {batch.keys()}")
                     continue
                labels = inputs.pop('label')

                outputs = model(**inputs)
                loss = criterion(outputs, labels)

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                # Get probabilities for ROC AUC (assuming binary/multiclass)
                if outputs.shape[1] > 1: # Softmax for multiclass/binary
                     probs = torch.softmax(outputs, dim=1)
                     # For binary, use prob of class 1; for multiclass, needs adjustment for one-vs-rest/all
                     if config['num_classes'] == 2:
                          all_probs.extend(probs[:, 1].cpu().numpy())
                     else: # Multiclass - store all probabilities for potential OvR AUC later
                          all_probs.extend(probs.cpu().numpy()) # Store as list of arrays

                elif outputs.shape[1] == 1: # Sigmoid for binary with 1 output node
                     probs = torch.sigmoid(outputs).squeeze()
                     all_probs.extend(probs.cpu().numpy())

            except Exception as e:
                print(f"\nError during evaluation batch: {e}")
                print(f"Batch keys: {batch.keys()}")
                continue

    avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0

    # Calculate ROC AUC
    roc_auc = None
    if config['num_classes'] == 2 and len(all_probs) == len(all_labels):
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError as e:
            # Common case: only one class present in labels
            if "Only one class present" in str(e):
                 print(f"Warning: Could not calculate ROC AUC - only one class present in this evaluation set.")
            else:
                 print(f"Warning: Could not calculate ROC AUC - {e}")
    elif config['num_classes'] > 2:
         print("ROC AUC calculation for multi-class not implemented here (requires OvR/OvO).")
         # You could calculate it here using sklearn.metrics.roc_auc_score(all_labels, all_probs, multi_class='ovr')
         # Make sure all_probs is shaped correctly [n_samples, n_classes]

    return avg_loss, accuracy, roc_auc, all_labels, all_preds


def run_training(config, model, train_dataloader, val_dataloader):
    """Main training loop, returns trained model and history."""
    device = config['device']
    # Optimizer - consider different optimizers based on config if needed
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    best_val_metric = -1.0 # Initialize for accuracy/AUC maximization
    metric_to_monitor = 'acc' # or 'roc_auc'
    # If monitoring loss: best_val_metric = float('inf')

    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_roc_auc': []
    }

    print("\n--- Starting Training ---")
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        # Training
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, device, config)
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)

        # Validation
        val_loss, val_acc, val_roc_auc, _, _ = evaluate_epoch(model, val_dataloader, criterion, device, config)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['val_roc_auc'].append(val_roc_auc) # Can be None

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}", end="")
        if val_roc_auc is not None:
            print(f", Val ROC AUC: {val_roc_auc:.4f}")
        else:
            print(" (Val ROC AUC: N/A)")

        # Determine current metric value for comparison
        current_val_metric = None
        if metric_to_monitor == 'acc':
            current_val_metric = val_acc
        elif metric_to_monitor == 'roc_auc':
            # Only use AUC if it was successfully calculated
            current_val_metric = val_roc_auc if val_roc_auc is not None else -1.0 # Fallback to -1 if AUC is None
        # Add elif for 'loss' if needed

        # Save best model
        if current_val_metric is not None and current_val_metric > best_val_metric:
             best_val_metric = current_val_metric
             save_model(model, config['model_save_path']) # Use save_model from utils
             print(f"  Best model saved (Epoch {epoch+1}, Val {metric_to_monitor}: {best_val_metric:.4f})")
        # Add condition for loss minimization if metric_to_monitor == 'loss'

    print("--- Training Finished ---")

    # Save final log and plot history
    save_training_log(training_history, config['log_file'])
    plot_training_history(training_history, config['history_plot_path'])

    return model, training_history