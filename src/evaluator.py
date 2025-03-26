import torch
import torch.nn as nn
from sklearn.metrics import classification_report
import numpy as np

# Import necessary functions/classes
from .trainer import evaluate_epoch # Reuse evaluate_epoch logic
from .utils import load_model_state, plot_confusion_matrix

def evaluate_model_on_test(config, model, test_dataloader, label_encoder):
    """Loads best model and evaluates it on the test set."""
    print("\n--- Evaluating on Test Set ---")
    device = config['device']
    criterion = nn.CrossEntropyLoss() # For calculating loss

    # Load the best model state
    if not load_model_state(model, config['model_save_path'], device):
         print("Failed to load best model for test evaluation. Aborting.")
         return

    # Evaluate using the same function as validation
    test_loss, test_acc, test_roc_auc, y_true, y_pred = evaluate_epoch(
        model, test_dataloader, criterion, device, config
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    if test_roc_auc is not None:
        print(f"Test ROC AUC: {test_roc_auc:.4f}")
    else:
        print("Test ROC AUC: N/A")

    # Ensure labels are available
    if not y_true or not y_pred:
        print("Evaluation did not produce labels or predictions. Cannot generate reports.")
        return

    # Classification Report
    try:
        target_names = label_encoder.classes_.astype(str)
        print("\nClassification Report:")
        # Handle potential UndefinedMetricWarning if some classes have no predictions
        # zero_division=0 means precision/recall/F1 become 0 for classes with no predicted samples.
        report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Could not generate classification report: {e}")

    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, target_names, config['cm_plot_path'])