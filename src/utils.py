import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from safetensors.torch import save_file, load_file

# Conditional import for torchviz
try:
    import torchviz
except ImportError:
    print("torchviz not found. Install it via 'pip install torchviz graphviz' for model visualization.")
    torchviz = None

# Import model classes from the models module
from . import models # Relative import

# --- Model Factory ---
def get_model(config):
    """Factory function to create model instance based on config."""
    model_type = config['model_type']
    print(f"Initializing model: {model_type}")

    # Use getattr for cleaner access, assuming class names match patterns
    model_class_name = ""
    if model_type == 'bert':
        model_class_name = 'BERTClassifier'
    elif model_type == 'lstm':
        model_class_name = 'LSTMClassifier'
    elif model_type == 'cnn':
        model_class_name = 'CNNClassifier'
    elif model_type == 'gcn':
        model_class_name = 'GCNClassifier'
    elif model_type == 'bert_lstm_gcn':
        model_class_name = 'BERTBiLSTMGCNModel'
    elif model_type == 'bert_lstm_cnn':
        model_class_name = 'BERTBiLSTMCnnModel'
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    try:
        ModelClass = getattr(models, model_class_name)
        model = ModelClass(config)
        # Ensure num_classes is set correctly after potential update in data loading
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
             if model.classifier.out_features != config['num_classes']:
                  print(f"Warning: Model classifier output size ({model.classifier.out_features}) "
                        f"differs from config num_classes ({config['num_classes']}). Check config/data.")
                  # Optionally, resize the classifier here, but it's usually better
                  # to ensure config is correct *before* model initialization.
                  # model.classifier = torch.nn.Linear(model.classifier.in_features, config['num_classes'])
    except AttributeError:
        raise ValueError(f"Model class '{model_class_name}' not found in models.py")

    return model.to(config['device'])

# --- Visualization ---
def visualize_model(model, dataloader, config):
    """Visualizes the model using torchviz."""
    if not torchviz:
        print("torchviz not installed. Skipping model visualization.")
        return
    if not dataloader or len(dataloader) == 0:
        print("Dataloader is empty. Skipping model visualization.")
        return

    filename = config['visualize_graph_path'] # Get path from config
    try:
        sample_batch = next(iter(dataloader))
        inputs = {k: v.to(config['device']) for k, v in sample_batch.items() if isinstance(v, torch.Tensor)}
        labels = inputs.pop('label', None) # Remove label

        # Ensure all required inputs for the specific model are present
        # This might need adjustment based on model specifics
        required_inputs = set()
        if 'bert' in config['model_type']: required_inputs.update(['input_ids', 'attention_mask'])
        if 'gcn' in config['model_type']: required_inputs.add('adj_matrix')
        if config.get('use_linguistic_features') and config['model_type'] in ['bert_lstm_gcn', 'bert_lstm_cnn']: required_inputs.add('linguistic_features')
        if config['model_type'] in ['lstm', 'cnn'] or (config['model_type'] == 'gcn' and 'bert' not in config['model_type']): required_inputs.add('token_ids')

        missing = required_inputs - set(inputs.keys())
        if missing:
             print(f"Warning: Missing inputs for visualization: {missing}. Trying anyway.")
             # Add dummy tensors if possible/necessary for visualization to proceed
             for key in missing:
                  if key == 'adj_matrix':
                       inputs[key] = torch.eye(config['max_length']).unsqueeze(0).to(config['device'])
                  elif key == 'linguistic_features':
                       inputs[key] = torch.zeros(1, config['linguistic_feat_dim']).to(config['device'])
                  elif key == 'token_ids':
                        inputs[key] = torch.zeros(1, config['max_length'], dtype=torch.long).to(config['device'])
                  # Add more dummy inputs if needed for other keys

        # Filter inputs to only those expected by the model's forward method
        # Inspecting signature is complex; simpler to rely on **kwargs in forward
        # Or ensure Dataset provides exactly what's needed.

        # Check if adj_matrix is needed but missing after potential dummy creation
        if 'gcn' in config['model_type'] and 'adj_matrix' not in inputs:
             print("Adj matrix needed for GCN visualization but missing. Skipping.")
             return

        y = model(**inputs)
        dot = torchviz.make_dot(y.mean(), params=dict(model.named_parameters())) # Use y.mean() for scalar output if needed
        dot.format = 'png'
        dot.render(filename.replace('.png', ''), cleanup=True) # cleanup removes intermediate files
        print(f"Model graph saved to {filename}")

    except Exception as e:
        print(f"Could not visualize model: {e}")
        import traceback
        traceback.print_exc()
        print("Check if sample batch inputs match the model's forward signature.")


# --- Plotting ---
def plot_training_history(history, filename):
    """Plots loss and accuracy curves."""
    if not history or not history.get('train_loss'):
        print("No training history found to plot.")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy & AUC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    if 'val_acc' in history and history['val_acc']:
        plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    # Plot AUC only if it exists and is not None for all epochs
    if 'val_roc_auc' in history and history['val_roc_auc'] and all(v is not None for v in history['val_roc_auc']):
        plt.plot(epochs, history['val_roc_auc'], 'go-', label='Validation ROC AUC')
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy / ROC AUC')
    plt.ylim(bottom=0) # Start y-axis at 0 for acc/auc
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Training history plot saved to {filename}")
    except Exception as e:
        print(f"Error saving training plot: {e}")
    plt.close() # Close the plot to free memory

def plot_confusion_matrix(y_true, y_pred, labels, filename):
    """Plots the confusion matrix."""
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(filename)
        print(f"Confusion matrix saved to {filename}")
        plt.close() # Close the plot
    except Exception as e:
        print(f"Could not generate or save confusion matrix: {e}")


# --- Saving & Logging ---
def save_training_log(history, filename):
    """Saves training history to a JSON file."""
    try:
        # Convert potential numpy types to native Python types for JSON serialization
        serializable_history = {}
        for key, values in history.items():
            serializable_history[key] = [float(v) if v is not None else None for v in values]

        with open(filename, 'w') as f:
            json.dump(serializable_history, f, indent=4)
        print(f"Training log saved to {filename}")
    except Exception as e:
        print(f"Error saving training log: {e}")

def save_model(model, filename):
    """Saves the model state dict using safetensors."""
    try:
        save_file(model.state_dict(), filename)
        print(f"Model state dict saved to {filename}")
    except Exception as e:
        print(f"Error saving model state dict: {e}")

def load_model_state(model, filename, device):
    """Loads the model state dict using safetensors."""
    if not os.path.exists(filename):
         print(f"Error: Model file not found at {filename}")
         return False
    try:
        state_dict = load_file(filename, device=device)
        model.load_state_dict(state_dict)
        model.to(device) # Ensure model is on the correct device
        print(f"Model state dict loaded from {filename}")
        return True
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return False