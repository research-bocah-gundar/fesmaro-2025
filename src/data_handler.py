import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import spacy
import os
import pickle
import hashlib
from sklearn.preprocessing import LabelEncoder

# --- Feature Extractor ---
class FeatureExtractor:
    # ... (Salin definisi kelas FeatureExtractor dari kode sebelumnya) ...
    # Pastikan __init__ menerima config, tokenizer, nlp
    # Pastikan metode menggunakan self.config, self.tokenizer, self.nlp
    def __init__(self, config, tokenizer, nlp=None):
        self.config = config
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.cache_dir = config['cache_dir']
        # Ensure cache dir exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, text, prefix):
        # Limit text length for hash to avoid issues with very long texts
        text_snippet = text[:2000]
        text_hash = hashlib.md5(text_snippet.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{prefix}_{text_hash}.pkl")

    def get_bert_inputs(self, text):
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config['max_length'],
            return_tensors='pt'
        )

    def get_dependency_graph(self, text, use_cache=True):
        if not self.nlp:
             adj = torch.eye(self.config['max_length'])
             return adj

        # Use limited text for cache key generation
        cache_file = self._get_cache_path(text, "dep_graph")

        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Cache load failed for {cache_file}. Recomputing. Error: {e}")

        doc = self.nlp(text[:1500]) # Limit SpaCy processing length
        max_len = self.config['max_length']
        adj_matrix = np.zeros((max_len, max_len))

        spacy_tokens = [token for token in doc if not token.is_space] # No slicing here yet

        # Build adjacency based on available tokens, up to max_len
        num_tokens_in_doc = len(spacy_tokens)
        effective_len = min(num_tokens_in_doc, max_len)

        for i in range(effective_len):
            token = spacy_tokens[i]
            # Check if head is within the effective length
            if token.head.i < num_tokens_in_doc:
                 head_idx_orig = token.head.i
                 # Map original head index to potentially truncated index
                 if head_idx_orig < effective_len:
                      adj_matrix[i, head_idx_orig] = 1
                      adj_matrix[head_idx_orig, i] = 1

        # Add self-loops up to max_len
        for i in range(max_len):
            adj_matrix[i, i] = 1 # Ensure all nodes including padding have self-loops

        # Normalize
        rowsum = np.array(adj_matrix.sum(1)).flatten()
        # Add epsilon to prevent division by zero before power operation
        rowsum[rowsum <= 0] = 1e-8
        d_inv_sqrt = np.power(rowsum, -0.5)
        # Handle potential NaNs/Infs AFTER power calculation
        d_inv_sqrt[np.isinf(d_inv_sqrt) | np.isnan(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        normalized_adj = d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt)

        result = torch.FloatTensor(normalized_adj)

        if use_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                print(f"Warning: Cache save failed for {cache_file}. Error: {e}")

        return result


    def get_linguistic_features(self, text):
        # Return zeros if nlp not available or feature not used
        if not self.nlp or not self.config.get('use_linguistic_features', False):
            return torch.zeros(self.config.get('linguistic_feat_dim', 9))

        features = {}
        text_limited = str(text)[:1000]
        words = text_limited.split()
        word_count = len(words)
        word_count_safe = word_count + 1e-6 # Avoid division by zero

        features['text_length'] = min(len(text_limited), 1000) / 1000
        features['word_count'] = min(word_count, 200) / 200
        features['avg_word_length'] = min(np.mean([len(w) for w in words]) if words else 0, 20) / 20
        features['exclamation_count'] = min(text_limited.count('!'), 10) / 10
        features['question_count'] = min(text_limited.count('?'), 10) / 10
        uppercase_word_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        features['uppercase_ratio'] = uppercase_word_count / word_count_safe

        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love',
                          'perfect', 'recommend', 'happy', 'awesome', 'like', 'highly'}
        negative_words = {'bad', 'poor', 'terrible', 'horrible', 'worst', 'waste', 'disappointed',
                          'disappointing', 'difficult', 'hate', 'problem', 'issue', 'fail', 'never', 'not'}

        lower_words = text_limited.lower().split()
        pos_count = sum(1 for w in lower_words if w in positive_words)
        neg_count = sum(1 for w in lower_words if w in negative_words)

        features['positive_word_count'] = min(pos_count, 20) / 20
        features['negative_word_count'] = min(neg_count, 20) / 20
        features['sentiment_polarity'] = (pos_count - neg_count) / word_count_safe # Normalized polarity

        # Ensure correct dimension
        feat_dim = self.config.get('linguistic_feat_dim', 9)
        feature_list = [
            features.get('text_length', 0),
            features.get('word_count', 0),
            features.get('avg_word_length', 0),
            features.get('exclamation_count', 0),
            features.get('question_count', 0),
            features.get('uppercase_ratio', 0),
            features.get('positive_word_count', 0),
            features.get('negative_word_count', 0),
            features.get('sentiment_polarity', 0)
            # Add more default features if feat_dim > 9
        ] + [0.0] * (feat_dim - 9) # Pad with zeros if needed

        return torch.FloatTensor(feature_list[:feat_dim]) # Truncate if needed


# --- Dataset ---
class SentimentDataset(Dataset):
    # ... (Salin definisi kelas SentimentDataset dari kode sebelumnya) ...
    # Pastikan __init__ menerima texts, labels, feature_extractor, config
    # Pastikan __getitem__ mengembalikan dict yang sesuai dengan kebutuhan model
     def __init__(self, texts, labels, feature_extractor, config):
        self.texts = texts
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.config = config
        self.model_type = config['model_type']
        self.use_linguistic = config.get('use_linguistic_features', False)

        # TODO: Implement non-BERT tokenization if needed
        # if self.model_type in ['lstm', 'cnn', 'gcn']:
        #     print("Warning: Non-BERT tokenization required but not implemented in Dataset.")
            # self.vocab = build_vocab(texts, config['vocab_size'])
            # self.tokenizer = simple_tokenizer or your specific tokenizer

     def __len__(self):
        return len(self.texts)

     def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] is not None else "" # Handle None/NaN
        label = self.labels[idx]

        item = {'label': torch.tensor(label, dtype=torch.long)}
        # Store text for potential debugging, but it won't be collated into a tensor
        # item['text'] = text

        # BERT-based features
        if 'bert' in self.model_type:
            bert_inputs = self.feature_extractor.get_bert_inputs(text)
            item['input_ids'] = bert_inputs['input_ids'].squeeze(0)
            item['attention_mask'] = bert_inputs['attention_mask'].squeeze(0)

        # GCN-specific features
        if 'gcn' in self.model_type:
             # Assumes GCN uses BERT embeddings if BERT is in model name
             # Otherwise, needs a different adj matrix based on non-BERT tokens
             if 'bert' in self.model_type or self.model_type == 'gcn': # Simplified: always use dep graph if gcn
                item['adj_matrix'] = self.feature_extractor.get_dependency_graph(text)
             # else: # Non-BERT GCN
                 # item['adj_matrix'] = self.feature_extractor.get_non_bert_adj(tokens) # Needs implementation

        # Linguistic features (only for specific models that use them)
        if self.use_linguistic and self.model_type in ['bert_lstm_gcn', 'bert_lstm_cnn']:
            item['linguistic_features'] = self.feature_extractor.get_linguistic_features(text)

        # Non-BERT token IDs (Needs implementation)
        if self.model_type in ['lstm', 'cnn'] or (self.model_type == 'gcn' and 'bert' not in self.model_type) :
             print(f"Warning: Requesting token_ids for {self.model_type}, but non-BERT tokenization is not implemented.")
             # Placeholder - replace with actual tokenization and padding
             item['token_ids'] = torch.zeros(self.config['max_length'], dtype=torch.long) # Dummy output

        # Filter out None values before returning, Pytorch Dataloader doesn't like None
        return {k: v for k, v in item.items() if v is not None}


# --- Data Loading Function ---
def load_data(config):
    """Loads train, validation, and test dataframes."""
    train_path = os.path.join(config['data_path'], config['train_file'])
    val_path = os.path.join(config['data_path'], config['val_file'])
    test_path = os.path.join(config['data_path'], config['test_file'])

    try:
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        # Rename and select columns (handle potential missing rename)
        text_col_orig = config.get('text_column_original', 'lemmatized_text') # Allow overriding original name
        text_col = config['text_column']
        label_col = config['label_column']

        for df in [df_train, df_val, df_test]:
            if text_col_orig in df.columns and text_col_orig != text_col:
                 df.rename(columns={text_col_orig: text_col}, inplace=True)
            if text_col not in df.columns:
                 raise KeyError(f"Text column '{text_col}' not found after potential rename.")
            if label_col not in df.columns:
                 raise KeyError(f"Label column '{label_col}' not found.")

        df_train = df_train[[label_col, text_col]].dropna()
        df_val = df_val[[label_col, text_col]].dropna()
        df_test = df_test[[label_col, text_col]].dropna()

        # Ensure text is string
        df_train[text_col] = df_train[text_col].astype(str)
        df_val[text_col] = df_val[text_col].astype(str)
        df_test[text_col] = df_test[text_col].astype(str)

        print(f"Loaded data: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

        # Encode labels
        label_encoder = LabelEncoder()
        df_train['label_encoded'] = label_encoder.fit_transform(df_train[label_col])
        df_val['label_encoded'] = label_encoder.transform(df_val[label_col])
        df_test['label_encoded'] = label_encoder.transform(df_test[label_col])

        # Dynamically update num_classes in config
        config['num_classes'] = len(label_encoder.classes_)
        print(f"Classes: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")
        print(f"Updated config['num_classes'] = {config['num_classes']}")


        return df_train, df_val, df_test, label_encoder

    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        raise
    except KeyError as e:
        print(f"Error accessing column: {e}. Check config 'text_column'/'label_column' and CSV headers.")
        raise

# --- DataLoader Creation Function ---
def create_dataloaders(config, df_train, df_val, df_test, feature_extractor):
    """Creates DataLoaders for train, val, and test sets."""

    text_col = config['text_column']
    # Use the encoded label column
    label_col_encoded = 'label_encoded'

    train_dataset = SentimentDataset(
        df_train[text_col].tolist(),
        df_train[label_col_encoded].tolist(),
        feature_extractor, config
    )
    val_dataset = SentimentDataset(
        df_val[text_col].tolist(),
        df_val[label_col_encoded].tolist(),
        feature_extractor, config
    )
    test_dataset = SentimentDataset(
        df_test[text_col].tolist(),
        df_test[label_col_encoded].tolist(),
        feature_extractor, config
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0 # Safer default for Windows
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0
    )

    print("DataLoaders created.")
    return train_dataloader, val_dataloader, test_dataloader

# --- Setup SpaCy and Tokenizer ---
def setup_nlp_tools(config):
    """Initializes SpaCy and BERT tokenizer."""
    nlp = None
    if config['model_type'] in ['gcn', 'bert_lstm_gcn'] or config.get('use_linguistic_features', False):
        try:
            nlp = spacy.load("en_core_web_sm")
            print("SpaCy model 'en_core_web_sm' loaded.")
        except OSError:
            print("Downloading spacy en_core_web_sm model...")
            try:
                 spacy.cli.download("en_core_web_sm")
                 nlp = spacy.load("en_core_web_sm")
                 print("SpaCy model 'en_core_web_sm' downloaded and loaded.")
            except Exception as e:
                 print(f"Failed to download or load spacy model: {e}")
                 print("GCN dependency graphs and linguistic features might not work.")

    tokenizer = None
    if 'bert' in config['model_type']:
        try:
            tokenizer = BertTokenizer.from_pretrained(config['bert_model_name'])
            print(f"BERT tokenizer '{config['bert_model_name']}' loaded.")
        except Exception as e:
            print(f"Failed to load BERT tokenizer '{config['bert_model_name']}': {e}")
            raise # Tokenizer is essential for BERT models

    # TODO: Add non-BERT tokenizer initialization if needed
    if config['model_type'] in ['lstm', 'cnn', 'gcn'] and 'bert' not in config['model_type']:
        print("Warning: Non-BERT tokenizer required but not implemented.")

    return tokenizer, nlp