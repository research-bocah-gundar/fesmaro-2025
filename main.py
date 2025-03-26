import argparse
import os

# Import functions/classes from our modules
from src.config_loader import load_config
from src.data_handler import load_data, setup_nlp_tools, FeatureExtractor, create_dataloaders
from src.utils import get_model, visualize_model
from src.trainer import run_training
from src.evaluator import evaluate_model_on_test

def main(args):
    # 1. Load Configuration
    config = load_config(args.model_type)

    # 2. Load Data & Encode Labels
    df_train, df_val, df_test, label_encoder = load_data(config)

    # 3. Setup NLP Tools (Tokenizer, SpaCy)
    tokenizer, nlp = setup_nlp_tools(config)

    # 4. Initialize Feature Extractor
    feature_extractor = FeatureExtractor(config, tokenizer, nlp)

    # 5. Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config, df_train, df_val, df_test, feature_extractor
    )

    # 6. Get Model Instance
    # Pass the updated config (with potentially changed num_classes)
    model = get_model(config)

    # 7. (Optional) Visualize Model Architecture
    if args.visualize:
        # Ensure dataloader is not empty before visualizing
        if train_loader and len(train_loader) > 0:
             visualize_model(model, train_loader, config)
        else:
             print("Skipping visualization: Training dataloader is empty.")


    # 8. Run Training
    # run_training returns the model (potentially modified, e.g., if layers were added)
    # and the history object
    trained_model, history = run_training(config, model, train_loader, val_loader)

    # 9. Evaluate Best Model on Test Set
    # Pass the model instance that was potentially modified during training/loading
    evaluate_model_on_test(config, trained_model, test_loader, label_encoder)

    print("\n--- Experiment Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sentiment Analysis Model Training")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['bert', 'lstm', 'cnn', 'gcn', 'bert_lstm_gcn', 'bert_lstm_cnn'],
        help="Type of model to train."
    )
    parser.add_argument(
        "--visualize",
        action='store_true', # Makes it a flag, e.g., --visualize
        help="Generate model architecture visualization (requires torchviz)."
    )
    # Add other command-line arguments if needed (e.g., override epochs, batch_size)

    args = parser.parse_args()
    main(args)