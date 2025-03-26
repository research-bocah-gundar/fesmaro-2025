import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoConfig
import numpy as np # Hanya untuk GraphConvolution reset (bisa dihapus jika init berbeda)

# --- GCN Layer ---
class GraphConvolution(nn.Module):
    # ... (Salin definisi kelas GraphConvolution dari kode sebelumnya) ...
    # Pastikan forward pass menangani batch
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        # input: [batch_size, seq_len, in_features]
        # adj: [batch_size, seq_len, seq_len]
        batch_size = input.size(0)
        weight_expanded = self.weight.unsqueeze(0).expand(batch_size, -1, -1)
        support = torch.bmm(input, weight_expanded) # [batch_size, seq_len, out_features]
        output = torch.bmm(adj, support) # [batch_size, seq_len, out_features]
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0).unsqueeze(0)
        return output

# --- Model Definitions ---
class BERTClassifier(nn.Module):
    # ... (Salin definisi kelas BERTClassifier dari kode sebelumnya) ...
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['bert_model_name'])
        self.bert_config_hf = AutoConfig.from_pretrained(config['bert_model_name']) # Renamed internal var
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(self.bert_config_hf.hidden_size, config['num_classes'])

        if config.get('freeze_bert', False): # Use .get for safety
            print("Freezing BERT parameters.")
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class LSTMClassifier(nn.Module):
    # ... (Salin definisi kelas LSTMClassifier dari kode sebelumnya) ...
    # Ingat: Model ini butuh token_ids (non-BERT) dari Dataset
     def __init__(self, config):
        super().__init__()
        self.config = config
        # Annoying fix for potential config load issue: ensure vocab_size is int
        vocab_size = int(config.get('vocab_size', 20000))
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'], padding_idx=0)
        self.lstm = nn.LSTM(config['embedding_dim'], config['hidden_dim'],
                            num_layers=config.get('lstm_layers', 1), bidirectional=True,
                            batch_first=True, dropout=config['dropout'] if config.get('lstm_layers', 1) > 1 else 0)
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(config['hidden_dim'] * 2, config['num_classes'])

     def forward(self, token_ids, **kwargs):
        if 'input_ids' in kwargs and token_ids is None: # Compatibility if passed incorrectly
             token_ids = kwargs['input_ids']
        if token_ids is None:
             raise ValueError("LSTMClassifier requires 'token_ids' input.")

        embedded = self.dropout(self.embedding(token_ids))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        return logits


class CNNClassifier(nn.Module):
    # ... (Salin definisi kelas CNNClassifier dari kode sebelumnya) ...
    # Ingat: Model ini butuh token_ids (non-BERT) dari Dataset
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = int(config.get('vocab_size', 20000))
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'], padding_idx=0)
        kernel_sizes = config.get('cnn_kernel_sizes', [3, 4, 5])
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=config['embedding_dim'],
                      out_channels=config['cnn_filters'],
                      kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(len(kernel_sizes) * config['cnn_filters'], config['num_classes'])

    def forward(self, token_ids, **kwargs):
        if 'input_ids' in kwargs and token_ids is None:
             token_ids = kwargs['input_ids']
        if token_ids is None:
             raise ValueError("CNNClassifier requires 'token_ids' input.")

        embedded = self.embedding(token_ids)
        embedded = embedded.permute(0, 2, 1)
        embedded = self.dropout(embedded)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        logits = self.classifier(cat)
        return logits

class GCNClassifier(nn.Module):
    # ... (Salin definisi kelas GCNClassifier dari kode sebelumnya) ...
    # Ingat: Model ini butuh token_ids dan adj_matrix (non-BERT) dari Dataset
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = int(config.get('vocab_size', 20000))
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'], padding_idx=0)
        self.dropout = nn.Dropout(config['dropout'])
        self.gcn_layers_list = nn.ModuleList() # Renamed internal var
        self.gcn_layers_list.append(GraphConvolution(config['embedding_dim'], config['hidden_dim']))
        for _ in range(1, config.get('gcn_layers', 2)):
            self.gcn_layers_list.append(GraphConvolution(config['hidden_dim'], config['hidden_dim']))
        self.classifier = nn.Linear(config['hidden_dim'], config['num_classes'])

    def forward(self, token_ids, adj_matrix, **kwargs):
        if 'input_ids' in kwargs and token_ids is None:
             token_ids = kwargs['input_ids']
        if token_ids is None or adj_matrix is None:
             raise ValueError("GCNClassifier requires 'token_ids' and 'adj_matrix' input.")

        embedded = self.dropout(self.embedding(token_ids))
        gcn_output = embedded
        for gcn_layer in self.gcn_layers_list:
            gcn_output = F.relu(gcn_layer(gcn_output, adj_matrix))
            gcn_output = self.dropout(gcn_output)
        pooled = torch.max(gcn_output, dim=1)[0]
        logits = self.classifier(pooled)
        return logits

class BERTBiLSTMGCNModel(nn.Module):
    # ... (Salin definisi kelas BERTBiLSTMGCNModel dari kode sebelumnya) ...
     def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['bert_model_name'])
        self.bert_config_hf = AutoConfig.from_pretrained(config['bert_model_name'])
        bert_dim = self.bert_config_hf.hidden_size
        hidden_dim = config['hidden_dim']

        if config.get('freeze_bert', False):
            print("Freezing BERT parameters.")
            for param in self.bert.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(bert_dim, hidden_dim, config.get('lstm_layers', 1),
                            bidirectional=True, batch_first=True,
                            dropout=config['dropout'] if config.get('lstm_layers', 1) > 1 else 0)

        self.gcn_layers_list = nn.ModuleList() # Renamed
        self.gcn_layers_list.append(GraphConvolution(bert_dim, hidden_dim))
        for _ in range(1, config.get('gcn_layers', 2)):
            self.gcn_layers_list.append(GraphConvolution(hidden_dim, hidden_dim))

        self.lstm_attention = nn.Linear(hidden_dim * 2, 1)
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gcn_proj = nn.Linear(hidden_dim, hidden_dim)

        self.use_linguistic = config.get('use_linguistic_features', False)
        fusion_input_dim = hidden_dim * 2 # Start with LSTM + GCN
        if self.use_linguistic:
            self.linguistic_proj = nn.Linear(config['linguistic_feat_dim'], hidden_dim)
            fusion_input_dim += hidden_dim

        self.fusion = nn.Linear(fusion_input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, config['num_classes'])
        self.dropout = nn.Dropout(config['dropout'])

     def forward(self, input_ids, attention_mask, adj_matrix, linguistic_features=None, **kwargs):
        # Check if required inputs are present
        if input_ids is None or attention_mask is None or adj_matrix is None:
             raise ValueError("BERTBiLSTMGCNModel requires 'input_ids', 'attention_mask', and 'adj_matrix'.")
        if self.use_linguistic and linguistic_features is None:
             # Provide default zeros if linguistic features are expected but not provided
             print("Warning: Linguistic features expected but not provided. Using zeros.")
             linguistic_features = torch.zeros(input_ids.size(0), self.config['linguistic_feat_dim'], device=input_ids.device)
             # raise ValueError("BERTBiLSTMGCNModel requires 'linguistic_features' when use_linguistic_features is True.")


        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        sequence_output_dropout = self.dropout(sequence_output)

        # LSTM Path
        lstm_output, _ = self.lstm(sequence_output_dropout)
        attn_weights = F.softmax(self.lstm_attention(lstm_output), dim=1)
        lstm_pooled = torch.sum(attn_weights * lstm_output, dim=1)
        lstm_features = F.relu(self.lstm_proj(lstm_pooled))
        lstm_features = self.dropout(lstm_features)

        # GCN Path
        gcn_output = sequence_output # Use original sequence output for GCN input
        for gcn_layer in self.gcn_layers_list:
            gcn_output = F.relu(gcn_layer(gcn_output, adj_matrix))
            gcn_output = self.dropout(gcn_output)
        gcn_pooled = torch.max(gcn_output, dim=1)[0]
        gcn_features = F.relu(self.gcn_proj(gcn_pooled))
        gcn_features = self.dropout(gcn_features)

        # Combine features
        combined = [lstm_features, gcn_features]
        if self.use_linguistic:
            ling_features_proj = F.relu(self.linguistic_proj(linguistic_features))
            ling_features_proj = self.dropout(ling_features_proj)
            combined.append(ling_features_proj)

        combined_cat = torch.cat(combined, dim=1)
        fused = F.relu(self.fusion(combined_cat))
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits


class BERTBiLSTMCnnModel(nn.Module):
    # ... (Salin definisi kelas BERTBiLSTMCnnModel dari kode sebelumnya) ...
     def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['bert_model_name'])
        self.bert_config_hf = AutoConfig.from_pretrained(config['bert_model_name'])
        bert_dim = self.bert_config_hf.hidden_size
        hidden_dim = config['hidden_dim']

        if config.get('freeze_bert', False):
            print("Freezing BERT parameters.")
            for param in self.bert.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(bert_dim, hidden_dim, config.get('lstm_layers', 1),
                            bidirectional=True, batch_first=True,
                            dropout=config['dropout'] if config.get('lstm_layers', 1) > 1 else 0)

        kernel_sizes = config.get('cnn_kernel_sizes', [3, 4, 5])
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=bert_dim,
                      out_channels=config['cnn_filters'],
                      kernel_size=k)
            for k in kernel_sizes
        ])
        cnn_output_dim = len(kernel_sizes) * config['cnn_filters']

        self.lstm_attention = nn.Linear(hidden_dim * 2, 1)
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cnn_proj = nn.Linear(cnn_output_dim, hidden_dim)

        self.use_linguistic = config.get('use_linguistic_features', False)
        fusion_input_dim = hidden_dim * 2 # LSTM + CNN
        if self.use_linguistic:
            self.linguistic_proj = nn.Linear(config['linguistic_feat_dim'], hidden_dim)
            fusion_input_dim += hidden_dim

        self.fusion = nn.Linear(fusion_input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, config['num_classes'])
        self.dropout = nn.Dropout(config['dropout'])

     def forward(self, input_ids, attention_mask, linguistic_features=None, **kwargs):
        if input_ids is None or attention_mask is None:
             raise ValueError("BERTBiLSTMCnnModel requires 'input_ids' and 'attention_mask'.")
        if self.use_linguistic and linguistic_features is None:
             print("Warning: Linguistic features expected but not provided. Using zeros.")
             linguistic_features = torch.zeros(input_ids.size(0), self.config['linguistic_feat_dim'], device=input_ids.device)
             # raise ValueError("BERTBiLSTMCnnModel requires 'linguistic_features' when use_linguistic_features is True.")


        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        sequence_output_dropout = self.dropout(sequence_output)

        # LSTM Path
        lstm_output, _ = self.lstm(sequence_output_dropout)
        attn_weights = F.softmax(self.lstm_attention(lstm_output), dim=1)
        lstm_pooled = torch.sum(attn_weights * lstm_output, dim=1)
        lstm_features = F.relu(self.lstm_proj(lstm_pooled))
        lstm_features = self.dropout(lstm_features)

        # CNN Path
        cnn_input = sequence_output.permute(0, 2, 1) # Use original sequence output
        conved = [F.relu(conv(cnn_input)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cnn_pooled = torch.cat(pooled, dim=1)
        cnn_features = F.relu(self.cnn_proj(cnn_pooled))
        cnn_features = self.dropout(cnn_features)

        # Combine features
        combined = [lstm_features, cnn_features]
        if self.use_linguistic:
            ling_features_proj = F.relu(self.linguistic_proj(linguistic_features))
            ling_features_proj = self.dropout(ling_features_proj)
            combined.append(ling_features_proj)

        combined_cat = torch.cat(combined, dim=1)
        fused = F.relu(self.fusion(combined_cat))
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits