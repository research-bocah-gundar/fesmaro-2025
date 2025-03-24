# Rencana Proyek Analisis Sentimen Amazon Reviews dengan Contextual-based GCN
**Durasi**: 24-30 Maret 2025 (7 hari)

## Timeline dan Target Harian

### Hari 1 (24 Maret 2025): Eksplorasi Data dan Persiapan Lingkungan
- **Pagi (08:00-12:00)**:
  - Unduh dataset Amazon Reviews
  - Exploratory Data Analysis (EDA)
    - Analisis distribusi kelas (positif vs negatif)
    - Statistik panjang review (title dan text)
    - Analisis kata-kata populer per kategori sentimen
    - Analisis missing values dan duplicates
  - Setup lingkungan pengembangan Python

- **Siang (13:00-17:00)**:
  - Persiapkan GitHub repository dengan struktur yang tepat
  - Dokumentasikan pemahaman permasalahan
  - Membuat notebook EDA dengan visualisasi awal
  - Implementasi fungsi data loader

- **Target**: Repository GitHub dengan EDA, visualisasi distribusi data, dan pemahaman dataset

### Hari 2 (25 Maret 2025): Preprocessing Data
- **Pagi (08:00-12:00)**:
  - Implementasi pipeline preprocessing:
    - Pembersihan teks (HTML, karakter khusus, tanda baca)
    - Case normalization
    - Tokenisasi (kata dan kalimat)
    - Lemmatization & stopword removal
    - Handling contractions dan spelling correction

- **Siang (13:00-17:00)**:
  - Feature extraction:
    - Bag-of-Words dan TF-IDF
    - Word embeddings preparation
    - Ekstraksi fitur leksikal (sentiment lexicon)
  - Split dataset (train, validation, test sets)
  - Validasi hasil preprocessing

- **Target**: Pipeline preprocessing lengkap dengan data yang siap untuk analisis topik dan modeling

### Hari 3 (26 Maret 2025): Implementasi LDA dan LSA
- **Pagi (08:00-12:00)**:
  - Implementasi Latent Semantic Analysis (LSA):
    - Membangun matriks dokumen-term
    - Implementasi TruncatedSVD untuk dimensionality reduction
    - Tuning parameter (jumlah komponen/topik)
    - Visualisasi dan interpretasi topik LSA

- **Siang (13:00-17:00)**:
  - Implementasi Latent Dirichlet Allocation (LDA):
    - Preprocessing khusus untuk LDA
    - Pemilihan jumlah topik optimal (coherence score)
    - Training model LDA
    - Visualisasi topik (pyLDAvis)
    - Interpretasi topik dan hubungannya dengan sentimen

- **Target**: Model topic modeling (LDA dan LSA) yang terlatih dengan visualisasi dan interpretasi topik

### Hari 4 (27 Maret 2025): Implementasi BERT+BiLSTM
- **Pagi (08:00-12:00)**:
  - Setup BERT untuk ekstraksi fitur kontekstual:
    - Persiapan BERT tokenizer
    - Konfigurasi model pretrained BERT
    - Implementasi extraction layer
    - Testing output representasi BERT

- **Siang (13:00-17:00)**:
  - Implementasi BiLSTM enhancement:
    - Arsitektur BiLSTM untuk contextual enhancement
    - Integrasi dengan output BERT
    - Setup training loop untuk fine-tuning
    - Validasi representasi kontekstual

- **Target**: Implementasi BERT+BiLSTM untuk representasi kontekstual selesai dengan validasi

### Hari 5 (28 Maret 2025): Implementasi Graph Convolutional Network
- **Pagi (08:00-12:00)**:
  - Implementasi konstruksi graph:
    - Definisi node (kata) dan edge (relasi)
    - Implementasi dependency parsing
    - Algoritma konstruksi graph berbasis similarity
    - Normalisasi adjacency matrix

- **Siang (13:00-17:00)**:
  - Implementasi model GCN:
    - Setup layer konvolusional graph
    - Implementasi pooling strategies
    - Integrasi dengan classifier
    - Testing forward pass model

- **Target**: Implementasi GCN lengkap dengan mekanisme konstruksi graph dan pooling strategy

### Hari 6 (29 Maret 2025): Training, Evaluasi, dan Dashboard - Bagian 1
- **Pagi (08:00-12:00)**:
  - Training dan optimasi model:
    - Full training pipeline
    - Hyperparameter tuning
    - Training dengan cross-validation
    - Saving checkpoint model terbaik

- **Siang (13:00-17:00)**:
  - Evaluasi komprehensif:
    - Kalkulasi metrik evaluasi
    - Confusion matrix dan classification report
    - Error analysis
    - Perbandingan dengan baseline models
  - Mulai implementasi Streamlit dashboard:
    - Setup struktur dasar dashboard
    - Komponen untuk upload data dan preprocessing

- **Target**: Model terlatih dengan evaluasi lengkap dan dasar dashboard Streamlit

### Hari 7 (30 Maret 2025): Dashboard dan Finalisasi
- **Pagi (08:00-12:00)**:
  - Finalisasi Streamlit dashboard:
    - Visualisasi hasil EDA
    - Visualisasi topik LDA dan LSA (interaktif)
    - Komponen analisis sentimen real-time
    - Visualisasi grafik hasil evaluasi
    - Visualisasi graph structure (node & edge)
    - Upload dan deployment dashboard

- **Siang (13:00-17:00)**:
  - Finalisasi dokumentasi:
    - Pembuatan dokumen metodologi (PDF)
    - Finalisasi README dan dokumentasi kode
    - Review akhir semua deliverables
    - Pengumpulan final melalui platform yang ditentukan

- **Target**: Dashboard Streamlit yang berfungsi penuh, dokumentasi lengkap, dan submission final

## Detail Implementasi Tahapan Proyek

### 1. Eksplorasi Data dan Preprocessing
- **EDA Komprehensif**:
  - Distribusi kelas sentimen (positif/negatif)
  - Statistik deskriptif untuk panjang teks
  - Word clouds untuk kata-kata umum di setiap sentimen
  - Analisis n-gram paling diskriminatif
  - Deteksi outlier (review sangat panjang/pendek)

- **Preprocessing Pipeline**:
  - Text cleaning: HTML removal, karakter khusus, emoji normalization
  - Tokenisasi: Menggunakan nltk dan spaCy
  - Normalisasi: Lowercase, lemmatization, contraction expansion
  - Custom preprocessing untuk BERT: WordPiece tokenization, special tokens
  - Custom preprocessing untuk LDA/LSA: Stopword removal lebih agresif, minimum word length

### 2. Topic Modeling dengan LDA dan LSA

#### LSA (Latent Semantic Analysis)
- **Implementasi**:
  - Representasi Vector Space Model dengan TF-IDF
  - Singular Value Decomposition dengan TruncatedSVD
  - Tuning jumlah komponen/topik (k = 5, 10, 15, 20)
  - Evaluasi hasil dekomposisi (explained variance)

- **Visualisasi dan Analisis**:
  - 2D visualization dengan PCA/t-SNE
  - Heatmap topics-terms
  - Analisis korelasi topik dengan sentimen

#### LDA (Latent Dirichlet Allocation)
- **Implementasi**:
  - Model training menggunakan gensim
  - Grid search parameter optimal:
    - Jumlah topik: 5-20
    - Alpha parameter: asymmetric vs symmetric
    - Beta parameter: auto vs fixed
  - Evaluasi coherence score untuk penentuan jumlah topik optimal

- **Visualisasi dan Analisis**:
  - Interactive visualization dengan pyLDAvis
  - Top words per topic dengan probabilitas
  - Topic distribution per dokumen
  - Korelasi topic distribution dengan label sentimen

### 3. Contextual Word Representations (BERT+BiLSTM)

#### BERT Implementation
- **Setup dan Konfigurasi**:
  - Model Base: BERT-base-uncased (12 layers, 768 hidden)
  - Tokenizer: BertTokenizer dengan padding dan truncation
  - Optimization: Gradual unfreezing (unfreeze layers secara bertahap)
  - Embedding extraction: Last 4 hidden layers concatenation

- **Fine-tuning Approach**:
  - Transfer learning dengan progressive layer unfreezing
  - Custom scheduler dengan warmup dan decay
  - Gradient accumulation untuk stabilitas

#### BiLSTM Enhancement
- **Arsitektur**:
  - Input: BERT embeddings (768 atau 3072 dimensi)
  - 2-layer Bidirectional LSTM
  - Hidden size: 256 per direction
  - Dropout: 0.2 untuk mencegah overfitting
  - Residual connections (output BERT + output BiLSTM)

### 4. Graph Convolutional Network

#### Graph Construction
- **Node Definition**:
  - Setiap kata dalam kalimat sebagai node
  - Feature node: Representasi kontekstual dari BERT+BiLSTM

- **Edge Construction Methods**:
  - **Dependency-based**: Menggunakan dependency parser dari spaCy
  - **Semantic**: Cosine similarity antar word representations
    - Threshold: 0.5 untuk filter edges
  - **Window-based**: Koneksi kata dalam window ±2
  - **Global-local**: Kombinasi local window dan semantic global connections

#### GCN Architecture
- **Layer Design**:
  - GCN Layer 1: Hidden units 256, ReLU, Layer Norm
  - GCN Layer 2: Hidden units 128, ReLU, Dropout 0.3
  - Spectral Graph Convolution dengan normalisasi Laplacian
  - Self-attention untuk pembobotan node

- **Pooling Strategies**:
  - Graph-level attention pooling
  - Hierarchical pooling dengan node clustering
  - Comparative analysis berbagai pooling methods

### 5. Model Training dan Evaluation

#### Training Pipeline
- **Optimizer Configuration**:
  - AdamW dengan learning rate 2e-5 untuk BERT layers
  - Adam dengan learning rate 1e-3 untuk GCN layers
  - Weight decay: 0.01 untuk regularization
  - Gradient clipping: 1.0

- **Training Schedule**:
  - Step 1: Train LDA/LSA untuk topic features
  - Step 2: Train BERT dengan frozen layers
  - Step 3: Fine-tune BERT+BiLSTM
  - Step 4: Train full GCN model dengan pretrained components

#### Evaluation Framework
- **Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC dan PR-AUC curves
  - Cohen's Kappa untuk inter-annotator agreement simulation
  - Log loss untuk probabilistic evaluation

- **Validation Strategy**:
  - 5-fold stratified cross-validation
  - Bootstrapping untuk confidence intervals
  - Learning curves untuk model convergence analysis

### 6. Streamlit Dashboard Development

#### Frontend Components
- **Data Exploration**:
  - Upload new dataset capability
  - Interactive visualizations dari EDA
  - Filter dan sorting capabilities

- **Topic Modeling Visualization**:
  - Interactive LDA visualization (pyLDAvis integration)
  - Topic-word distribution heatmaps
  - Document-topic similarity explorer

- **Sentiment Analysis**:
  - Real-time sentiment prediction untuk input user
  - Confidence score visualization
  - Interpretability tools (highlighting influential words)

- **Model Performance**:
  - Interactive confusion matrix
  - ROC curves dengan comparison view
  - Accuracy breakdown by text length/complexity

#### Backend Integration
- **Model Serving**:
  - Preloaded trained models dengan caching
  - Batch prediction capability
  - API endpoint untuk model predictions
  - Streaming processing untuk large datasets

- **Data Processing Pipeline**:
  - Preprocessing workflow terintegrasi
  - Progress indicators untuk long-running tasks
  - Error handling dan validation checks

### 7. Final Deliverables dan Documentation

#### Technical Documentation
- **Methodology Paper**:
  - Introduction & background
  - Related work & literature review
  - Detailed methodology (LDA/LSA, BERT+BiLSTM, GCN)
  - Implementation details & architecture
  - Experimental results & analysis
  - Conclusion & future work

- **Code Documentation**:
  - Function-level docstrings (Google style)
  - Architecture diagrams
  - Class hierarchy & dependency graphs
  - Usage examples dan API reference

#### Repository Organization
```
amazon-reviews-sentiment/
├── README.md                      # Deskripsi proyek dan petunjuk
├── data/
│   ├── raw/                       # Data mentah Amazon Reviews
│   ├── processed/                 # Data yang telah diproses
│   └── embeddings/                # Pretrained embeddings & model weights
├── notebooks/
│   ├── 1_EDA.ipynb               # Exploratory Data Analysis
│   ├── 2_Preprocessing.ipynb     # Text preprocessing pipeline
│   ├── 3_Topic_Modeling.ipynb    # LDA dan LSA implementation
│   ├── 4_BERT_BiLSTM.ipynb       # Contextual representations
│   ├── 5_GCN_Implementation.ipynb # Graph construction & GCN
│   └── 6_Evaluation.ipynb        # Model evaluation & comparison
├── src/
│   ├── preprocessing/             # Preprocessing modules
│   │   ├── cleaner.py            # Text cleaning utilities
│   │   ├── tokenizer.py          # Tokenization utilities
│   │   └── feature_extraction.py # Feature extraction methods
│   ├── modeling/                  # Model implementation
│   │   ├── topic_models.py       # LDA dan LSA implementation
│   │   ├── bert_bilstm.py        # BERT+BiLSTM implementation
│   │   ├── graph_construction.py # Graph construction utilities
│   │   ├── gcn.py                # GCN implementation
│   │   └── classifier.py         # Final classifier implementation
│   ├── evaluation/                # Evaluation utilities
│   │   ├── metrics.py            # Evaluation metrics
│   │   ├── visualization.py      # Evaluation visualization
│   │   └── error_analysis.py     # Error analysis utilities
│   └── utils/                     # Utility functions
│       ├── data_utils.py         # Data handling utilities
│       ├── training_utils.py     # Training helper functions
│       └── visualization_utils.py # Visualization utilities
├── streamlit/                     # Streamlit dashboard
│   ├── app.py                    # Main application
│   ├── pages/                    # Multi-page components
│   │   ├── eda.py               # EDA visualization
│   │   ├── topic_modeling.py    # Topic modeling visualization
│   │   ├── sentiment_analysis.py # Sentiment prediction
│   │   └── model_evaluation.py  # Model performance visualization
│   └── components/               # Reusable UI components
├── docs/                          # Documentation
│   ├── methodology.pdf           # Methodology paper
│   ├── presentation.pdf          # Presentation slides
│   └── installation.md           # Installation guide
├── models/                        # Saved models
│   ├── lda/                      # LDA model files
│   ├── lsa/                      # LSA model files
│   ├── bert_bilstm/              # BERT+BiLSTM checkpoints
│   └── gcn/                      # GCN model checkpoints
├── requirements.txt               # Dependensi proyek
└── setup.py                       # Package installation script
```

Dengan rencana detail ini, Anda memiliki kerangka kerja komprehensif untuk mengimplementasikan proyek analisis sentimen Amazon Reviews yang mencakup:
1. Topic modeling dengan LDA dan LSA untuk mengekstrak struktur tersembunyi
2. Representasi kontekstual kata dengan BERT+BiLSTM
3. Graph neural network untuk menangkap relasi struktural
4. Dashboard interaktif dengan Streamlit untuk visualisasi dan analisis hasil