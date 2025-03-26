import json
import os
import torch

def load_config(model_type, config_dir="configs"):
    """Loads base config and merges model-specific config."""
    base_config_path = os.path.join(config_dir, "base_config.json")
    model_config_path = os.path.join(config_dir, f"{model_type}.json")

    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config not found at {base_config_path}")
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config not found at {model_config_path}")

    with open(base_config_path, 'r') as f:
        config = json.load(f)
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    # Merge dictionaries, model_config overrides base_config
    config.update(model_config)

    # --- Post-processing ---
    # Set device
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Construct full paths
    config['model_save_path'] = os.path.join(config['log_dir'], config['model_save_name'])
    config['log_file'] = os.path.join(config['log_dir'], config['log_file_name'])
    config['history_plot_path'] = os.path.join(config['log_dir'], config['history_plot_name'])
    config['cm_plot_path'] = os.path.join(config['log_dir'], config['cm_plot_name'])
    config['visualize_graph_path'] = os.path.join(config['log_dir'], config['visualize_graph_name'].format(model_type=model_type))

    # Ensure log/cache directories exist
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['cache_dir'], exist_ok=True)

    print(f"--- Configuration for model '{model_type}' ---")
    for key, value in config.items():
        print(f"- {key}: {value}")
    print("--------------------------------------------")

    return config