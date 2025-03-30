# Analisis Sentimen Ulasan Amazon Menggunakan Model Deep Learning Hybrid

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Ganti dengan lisensi Anda jika berbeda -->

Repositori ini berisi kode dan dokumentasi untuk proyek analisis sentimen pada dataset ulasan Amazon. Proyek ini mengeksplorasi dan membandingkan berbagai arsitektur deep learning, termasuk model dasar (CNN, LSTM, GCN, BERT) dan model hybrid (BERT-LSTM-CNN, BERT-LSTM-GCN), untuk tugas klasifikasi sentimen biner (positif/negatif). Eksperimen dicatat dan dimonitor menggunakan Weights & Biases (WandB), dan hasil akhirnya disajikan dalam dashboard interaktif yang dibangun dengan Streamlit.

## Daftar Isi

- [Analisis Sentimen Ulasan Amazon Menggunakan Model Deep Learning Hybrid](#analisis-sentimen-ulasan-amazon-menggunakan-model-deep-learning-hybrid)
  - [Daftar Isi](#daftar-isi)
  - [1. Ringkasan Proyek](#1-ringkasan-proyek)
  - [2. Fitur Utama](#2-fitur-utama)
  - [3. Dataset](#3-dataset)
  - [4. Metodologi](#4-metodologi)
    - [4.1. Pemuatan dan Pra-pemrosesan Data](#41-pemuatan-dan-pra-pemrosesan-data)
    - [4.2. Analisis Data Eksploratif (EDA)](#42-analisis-data-eksploratif-eda)
    - [4.3. Rekayasa Fitur (Feature Engineering)](#43-rekayasa-fitur-feature-engineering)
    - [4.4. Pemodelan Topik (Topic Modeling) - Eksplorasi](#44-pemodelan-topik-topic-modeling---eksplorasi)
    - [4.5. Pelatihan dan Evaluasi Model](#45-pelatihan-dan-evaluasi-model)
    - [4.6. Pelacakan Eksperimen (WandB)](#46-pelacakan-eksperimen-wandb)
    - [4.7. Deployment Model dan Dashboard](#47-deployment-model-dan-dashboard)
  - [5. Teknologi yang Digunakan](#5-teknologi-yang-digunakan)
  - [6. Struktur Direktori](#6-struktur-direktori)
  - [7. Hasil dan Laporan Pelatihan](#7-hasil-dan-laporan-pelatihan)
  - [8. Demo Langsung (Dashboard Streamlit)](#8-demo-langsung-dashboard-streamlit)
  - [9. Pengaturan dan Penggunaan](#9-pengaturan-dan-penggunaan)
    - [9.1. Prasyarat](#91-prasyarat)
    - [9.2. Instalasi](#92-instalasi)
    - [9.3. Menjalankan Pelatihan](#93-menjalankan-pelatihan)
  - [10. Lisensi](#10-lisensi)

## 1. Ringkasan Proyek

Proyek ini bertujuan untuk:
1.  Melakukan pra-pemrosesan dan analisis eksploratif pada dataset ulasan Amazon berskala besar.
2.  Mengimplementasikan dan melatih berbagai model deep learning untuk klasifikasi sentimen, termasuk:
    *   Model Dasar: CNN, LSTM, GCN, BERT (fine-tuning)
    *   Model Hybrid: BERT-LSTM-CNN, BERT-LSTM-GCN
3.  Memanfaatkan fitur linguistik tambahan dan graf dependensi sintaksis untuk meningkatkan performa model.
4.  Menggunakan Weights & Biases (WandB) secara ekstensif untuk melacak eksperimen, memvisualisasikan metrik, menyimpan artefak model, dan menghasilkan laporan pelatihan yang komprehensif.
5.  Mendeploy model terlatih dan visualisasi data dalam sebuah dashboard interaktif menggunakan Streamlit.

## 2. Fitur Utama

*   **Pra-pemrosesan Data Komprehensif**: Pipeline pembersihan teks yang detail (lowercase, HTML removal, ekspansi kontraksi, URL/email/mention removal, normalisasi angka/ID produk, penghapusan tanda baca, penghapusan stopwords dengan preservasi negasi, lemmatisasi).
*   **Analisis Data Eksploratif (EDA)**: Analisis distribusi kelas, panjang teks, jumlah kata, deteksi outlier, dan visualisasi menggunakan word clouds.
*   **Rekayasa Fitur Fleksibel**: Ekstraksi fitur berbasis teks (panjang, jumlah kata, dll.), fitur spesifik sentimen, dan TF-IDF (n-grams).
*   **Dukungan Berbagai Model**: Implementasi modular untuk CNN, LSTM, GCN, BERT, dan kombinasi hybrid.
*   **Pelacakan Eksperimen dengan WandB**: Integrasi penuh dengan WandB untuk logging metrik, hyperparameter, konfigurasi, visualisasi, penyimpanan artefak (model `.safetensors`), dan pembuatan laporan.
*   **Dukungan Multi-GPU**: Pemanfaatan `torch.nn.DataParallel` untuk mempercepat pelatihan pada environment multi-GPU.
*   **Penyimpanan Model Efisien**: Menggunakan format `.safetensors` untuk menyimpan bobot model.
*   **Deployment dengan Streamlit**: Dashboard interaktif untuk visualisasi EDA, pemodelan topik LDA, evaluasi model, dan pengujian klasifikasi sentimen secara real-time.

## 3. Dataset

Dataset yang digunakan adalah **Amazon Reviews** yang tersedia di Kaggle:
*   **Sumber**: [kritanjalijain/amazon-reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
*   **Deskripsi**: Dataset ini berisi jutaan ulasan produk dari Amazon. Untuk proyek ini, fokusnya adalah pada teks ulasan (`title` dan `text` digabung) dan label sentimennya.
*   **Label**: Awalnya label adalah 1 (negatif) dan 2 (positif). Ini dipetakan ke representasi biner: 0 (negatif) dan 1 (positif).
*   **Ukuran**: Dataset asli sangat besar (3.6 juta ulasan latih). Karena alasan komputasi, dilakukan sampling setelah filtering awal berdasarkan kuartil panjang teks/jumlah kata.
    *   **Data Latih (setelah sampling)**: 100.000 ulasan (50k positif, 50k negatif)
    *   **Data Validasi (setelah sampling)**: 20.000 ulasan (10k positif, 10k negatif)
    *   **Data Uji (setelah sampling)**: 30.000 ulasan (15k positif, 15k negatif)

## 4. Metodologi

Proses kerja proyek ini meliputi beberapa tahapan utama:

### 4.1. Pemuatan dan Pra-pemrosesan Data

1.  **Pemuatan Data**: Memuat dataset `train.csv` dan `test.csv` menggunakan Pandas.
2.  **Pembersihan Awal**: Penamaan kolom (`label`, `title`, `text`), pemetaan label ke format biner (0/1), dan penggabungan kolom `title` dan `text`.
3.  **Filtering & Sampling**: Karena ukuran data yang sangat besar, data difilter terlebih dahulu untuk mempertahankan ulasan dengan panjang teks dan jumlah kata antara kuartil pertama (Q1) dan ketiga (Q3) yang dihitung pada EDA. Kemudian dilakukan sampling acak yang seimbang untuk membuat set data latih (100k), validasi (20k), dan uji (30k) yang lebih manageable.
4.  **Pipeline Pembersihan Teks Lanjutan (pada data sampel)**:
    *   Konversi ke huruf kecil.
    *   Penghapusan tag HTML.
    *   Ekspansi kontraksi (e.g., "don't" -> "do not").
    *   Penghapusan URL, alamat email, dan mention pengguna (@user).
    *   Penggantian nomor/ID produk panjang dengan token `product`.
    *   Penggantian angka dengan token `number`.
    *   Penghapusan tanda baca.
    *   Penghapusan spasi berlebih.
    *   Penghapusan stopwords bahasa Inggris, **kecuali** kata negasi ('no', 'not', 'never', dll.) yang penting untuk analisis sentimen.
    *   Lemmatisasi menggunakan NLTK `WordNetLemmatizer`.
    *   Hasil disimpan dalam kolom `lemmatized_text`.

### 4.2. Analisis Data Eksploratif (EDA)

Dilakukan pada data latih *sebelum* sampling (untuk pemahaman data asli) dan *setelah* sampling (pada data yang digunakan untuk model).
*   **Distribusi Kelas**: Dataset asli dan sampel sangat seimbang (50/50 antara kelas positif dan negatif).
*   **Panjang Teks & Jumlah Kata**: Distribusi cenderung miring ke kanan (right-skewed), menunjukkan mayoritas ulasan relatif pendek, namun ada ekor panjang ulasan yang sangat panjang. Ulasan negatif cenderung sedikit lebih panjang daripada ulasan positif.
*   **Deteksi Outlier**: Menggunakan metode IQR. Tidak ada outlier signifikan pada panjang teks, namun terdeteksi beberapa outlier pada jumlah kata (ulasan dengan jumlah kata sangat tinggi). Outlier ini efektif dihilangkan saat filtering data sebelum sampling.
*   **Word Clouds**: Visualisasi kata-kata yang paling sering muncul untuk sentimen positif dan negatif setelah pra-pemrosesan, memberikan insight awal tentang kata kunci pembeda sentimen.

### 4.3. Rekayasa Fitur (Feature Engineering)

Fitur tambahan dibuat dari teks (sebelum dan sesudah pembersihan) untuk memperkaya representasi data, terutama untuk model non-BERT dan hybrid:
*   **Fitur Berbasis Teks**: `text_length`, `word_count`, `avg_word_length`, `exclamation_count`, `question_count`, `uppercase_word_count`, `uppercase_ratio`.
*   **Fitur Spesifik Sentimen**: `positive_word_count`, `negative_word_count`, `sentiment_word_ratio` (berdasarkan daftar kata kunci positif/negatif sederhana).
*   **TF-IDF**: Dihasilkan dari `lemmatized_text` menggunakan `TfidfVectorizer` (sklearn) dengan parameter `min_df=100`, `max_df=0.75`, `ngram_range=(1, 2)`, dan `max_features=10000`. Fitur dan vectorizer disimpan menggunakan pickle.
*   **Graf Dependensi (untuk GCN)**: Dibuat menggunakan SpaCy (`en_core_web_sm`) untuk memodelkan hubungan sintaksis antar kata. Matriks adjacency dinormalisasi secara simetris. Proses ini di-cache untuk efisiensi.

### 4.4. Pemodelan Topik (Topic Modeling) - Eksplorasi

Sebagai analisis tambahan, Latent Dirichlet Allocation (LDA) dan Latent Semantic Analysis (LSA) diterapkan pada `lemmatized_text` dari data sampel (100k) menggunakan Gensim.
*   **Tujuan**: Mengidentifikasi topik-topik tersembunyi dalam ulasan.
*   **Metode**: LDA dilatih pada representasi Bag-of-Words (BoW), LSA pada TF-IDF. Jumlah topik optimal untuk LDA ditentukan menggunakan Coherence Score (C_v), hasilnya 15 topik.
*   **Hasil**: LDA menghasilkan topik yang lebih koheren dan terinterpretasi dibandingkan LSA. Hasil pemodelan topik LDA divisualisasikan menggunakan `pyLDAvis` dan diintegrasikan ke dalam dashboard Streamlit. Fitur distribusi topik per dokumen (probabilitas topik untuk LDA, koordinat topik untuk LSA) juga diekstrak.

### 4.5. Pelatihan dan Evaluasi Model

*   **Framework**: PyTorch.
*   **Model yang Dilatih**: CNN, LSTM (BiLSTM), GCN, BERT (bert-base-uncased fine-tuning), BERT-LSTM-CNN, BERT-LSTM-GCN.
*   **Input**:
    *   BERT & Hybrid: `input_ids`, `attention_mask` dari tokenizer `BertTokenizer`.
    *   Non-BERT: `token_ids` dari vocabulary kustom (dibangun dari data latih, `max_vocab_size=10000`, `min_freq=2`), dipad/truncated ke `max_length`.
    *   GCN & Hybrid (GCN): Matriks adjacency `adj_matrix` dari graf dependensi.
    *   Hybrid (Linguistik): Fitur linguistik tambahan jika `use_linguistic_features=True`.
*   **Proses Pelatihan**:
    *   Loop pelatihan standar dengan validasi per epoch.
    *   Optimizer: AdamW.
    *   Loss Function: CrossEntropyLoss.
    *   Early Stopping: Berdasarkan metrik validasi (akurasi atau ROC AUC) dengan patience=3.
    *   Penyimpanan Model Terbaik: Bobot model dengan performa validasi terbaik disimpan menggunakan `safetensors.torch.save_file`.
    *   Dukungan Multi-GPU: Menggunakan `nn.DataParallel` jika >1 GPU terdeteksi.

### 4.6. Pelacakan Eksperimen (WandB)

*   Setiap run pelatihan diinisialisasi dan dilacak menggunakan WandB.
*   **Logging**: Hyperparameter, konfigurasi, metrik training/validation (loss, accuracy, ROC AUC) per epoch, metrik sistem.
*   **Artefak**: Model terbaik (`.safetensors`) disimpan sebagai artefak WandB. Plot history pelatihan dan confusion matrix juga di-log sebagai gambar/artefak.
*   **Laporan**: Laporan pelatihan detail dihasilkan di WandB untuk setiap model, memvisualisasikan progres metrik dan perbandingan. (Link laporan disediakan di bawah).

### 4.7. Deployment Model dan Dashboard

*   **Teknologi**: Streamlit, dideploy ke Streamlit Cloud.
*   **Bobot Model**: Bobot model terbaik (format `.safetensors`) dari hasil training di WandB diunduh, kemudian diunggah ke Virtual Private Server (VPS) agar dapat diakses secara publik oleh aplikasi Streamlit.
*   **Dashboard**: Aplikasi Streamlit dengan 4 halaman utama:
    1.  **Exploratory Data Analysis (EDA)**: Menampilkan visualisasi hasil EDA (distribusi kelas, panjang teks, dll.).
    2.  **LDA Visualization**: Menampilkan visualisasi interaktif topik LDA menggunakan `pyLDAvis`.
    3.  **Model Evaluation**: Menampilkan ringkasan performa dan perbandingan antar model (berdasarkan hasil evaluasi test set).
    4.  **Sentiment Classifier Test**: Memungkinkan pengguna memasukkan teks ulasan baru dan mendapatkan prediksi sentimen dari model-model yang telah dilatih (model dimuat dari VPS).
*   **Akses**: Dashboard dapat diakses publik melalui URL yang disediakan.

## 5. Teknologi yang Digunakan

Kami menggunakan teknlogi sebagai berikut
<p align="left">
  <img height="64" src="https://cdn.simpleicons.org/kaggle" alt="Kaggle"/>
  <img height="64" src="https://cdn.simpleicons.org/googlecolab" alt="Google Colab"/>

  <img height="64" src="https://cdn.simpleicons.org/python" alt="Python"/>
  <img height="64" src="https://cdn.simpleicons.org/anaconda" alt="Anaconda"/>
  <img height="64" src="https://cdn.simpleicons.org/jupyter" alt="Jupyter"/>
  <img height="64" src="https://cdn.simpleicons.org/streamlit" alt="Streamlit"/>
  <img height="64" src="https://cdn.simpleicons.org/pytorch" alt="PyTorch"/>
  <img height="64" src="https://cdn.simpleicons.org/pandas" alt="Pandas"/>
</p>

*   **Bahasa**: Python 3
*   **Analisis Data & Pra-pemrosesan**: Pandas, NumPy, NLTK, Scikit-learn, Contractions, RE, Collections
*   **Machine Learning / Deep Learning**: PyTorch, Transformers (Hugging Face), Safetensors
*   **NLP Lanjutan**: SpaCy (untuk graf dependensi), Gensim (untuk Topic Modeling)
*   **Visualisasi**: Matplotlib, Seaborn, WordCloud, pyLDAvis, Torchviz (opsional)
*   **Pelacakan Eksperimen**: Weights & Biases (WandB)
*   **Deployment**: Streamlit, Streamlit-Option-Menu, TextStat
*   **Utilitas**: OS, Pickle, Hashlib, JSON, TQDM, Gdown

## 6. Struktur Direktori

Struktur direktori proyek diorganisir sebagai berikut untuk memisahkan data, konfigurasi, kode sumber, model, dan aset lainnya:
```
|   .gitignore
|   fesmaro-2025-analisis-big-data-lsalda.ipynb
|   fesmaro-2025-analisis-big-data-training.ipynb
|   fesmaro-2025-analisis-big-data.ipynb
|   lda_visualization.html
|   LICENSE
|   main.py
|   README.md
|   requirements.txt
|   structure.txt
|   tfidf_features.pkl
|   
+---assets
|   +---img
|   |   +---eda
|   |   |       deteksi-outlier(char).png
|   |   |       deteksi-outlier(word).png
|   |   |       distribution-of-review-length(char).png
|   |   |       distribution-of-review-sentiment(word).png
|   |   |       distribution-of-review-sentiment.png
|   |   |       negative-reviews-word.png
|   |   |       positive-reviews-word.png
|   |   |       
|   |   \---model
|   |       +---bert
|   |       |       bert-conf-matrix.png
|   |       |       bert-graph.png
|   |       |       bert-training-curves.png
|   |       |       
|   |       +---bert-lstm-cnn
|   |       |       bert-lstm-cnn-conf-matrix.png
|   |       |       bert-lstm-cnn-graph.png
|   |       |       bert-lstm-cnn-training-curves.png
|   |       |       
|   |       +---bert-lstm-gcn
|   |       |       bert-lstm-gcn-conf-matrix.png
|   |       |       bert-lstm-gcn-graph.png
|   |       |       bert-lstm-gcn-training-curves.png
|   |       |       
|   |       +---cnn
|   |       |       cnn-conf-matrix.png
|   |       |       cnn-graph.png
|   |       |       cnn-training-curves.png
|   |       |       
|   |       +---gcn
|   |       |       gcn-conf-matrix.png
|   |       |       gcn-graph.png
|   |       |       gcn-training-curves.png
|   |       |       
|   |       \---lstm
|   |               lstm-conf-matrix.png
|   |               lstm-graph.png
|   |               lstm-training-curves.png
|   |               
|   \---Topic Modelling
|           LDA-score.png
|           
+---configs
|       base_config.json
|       bert.json
|       bert_lstm_cnn.json
|       bert_lstm_gcn.json
|       cnn.json
|       gcn.json
|       lstm.json
|       README.md
|       
+---data
|       df_test.csv
|       df_val.csv
|       df_with_lda_features.csv
|       df_with_lsa_features.csv
|       final_df.csv
|       README.md
|       
+---model
|       cnn_best_model.safetensors
|       gcn_best_model.safetensors
|       lstm_best_model.safetensors
|       
+---src
|       config_loader.py
|       data_handler.py
|       evaluator.py
|       models.py
|       trainer.py
|       utils.py
|       __init__.py
|       
\---timeline
        README.md
        timeline.html
```

## 7. Hasil dan Laporan Pelatihan

Pelatihan untuk setiap model dicatat secara detail menggunakan Weights & Biases. Laporan lengkap yang mencakup kurva pembelajaran (loss, akurasi, ROC AUC), konfigurasi, dan metrik sistem dapat diakses melalui link berikut:

*(Penting: Beberapa link mungkin memerlukan token akses yang disertakan dalam URL untuk dilihat publik. Berhati-hatilah saat membagikan link dengan token akses.)*

📍 **Laporan Model WandB** 📍

*   **BERT-LSTM-CNN**: [Training Report](https://wandb.ai/masriq/sentiment-analysis-hybrid/reports/Training-Report--VmlldzoxMjAyMzYwNg?accessToken=prnuymr17257syith3higdj2me13miwk61h975fnhcodc7iptdao7nbjeq4f9r4q)
*   **BERT-LSTM-GCN**: [Training Report](https://api.wandb.ai/links/masriq/dnk86q9m)
*   **CNN**: [Training Report](https://wandb.ai/hkacode/sentiment-analysis-hybrid/reports/Training-Report--VmlldzoxMjAzNDE2MQ?accessToken=z70w4fkedok8jmn8nj8qho1fzwoycsysizlg1rb77inaizz0hd4aez4ih2pbomz6)
*   **LSTM**: [Training Report](https://api.wandb.ai/links/hkacode/xh7y704a)
*   **GCN**: [Training Report](https://api.wandb.ai/links/hkacode/o5ef19js)
*   **BERT**: [Training Report](https://wandb.ai/hkacode/sentiment-analysis-hybrid/reports/Model-BERT-Training-Report--VmlldzoxMjAzNTI0OQ?accessToken=zy1yhd8a1gqvznutyzpjacmq1enqf8yzg09t5weafu5gikozipngbimebdfh3l3h)

**Ringkasan Performa (pada Test Set - Model Terbaik):**

| Model           | Test Accuracy | Test ROC AUC | Catatan                                     |
| :-------------- | :-----------: | :----------: | :------------------------------------------ |
| CNN             |    86.02%     |    0.9386    | Performa baseline solid, seimbang.          |
| LSTM            |    82.87%     |    0.9393    | Overfitting, bias ke kelas positif.         |
| GCN             |    84.99%     |    0.9302    | Sedikit lebih baik dari LSTM, sedikit bias. |
| **BERT**        |  **94.65%**   |  **0.9867**  | **Performa terbaik**, sangat seimbang.      |
| BERT-LSTM-CNN   |    94.49%     |    0.9866    | Mirip BERT, tidak ada peningkatan signifikan. |
| BERT-LSTM-GCN   |    94.30%     |    0.9852    | Mirip BERT, tidak ada peningkatan signifikan. |

*   Model **BERT** yang di-fine-tune menunjukkan dominasi yang jelas dibandingkan model non-BERT dan hybrid pada dataset ini.
*   Model hybrid tidak memberikan peningkatan performa yang signifikan dibandingkan BERT murni, menunjukkan representasi BERT sudah sangat kuat.

## 8. Demo Langsung (Dashboard Streamlit)

Dashboard interaktif yang menampilkan hasil EDA, visualisasi LDA, evaluasi model, dan memungkinkan pengujian klasifikasi sentimen secara langsung telah dideploy menggunakan Streamlit Cloud.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rbg-sentiment-analysis.streamlit.app/)

**Akses Dashboard:** 👉 [**https://rbg-sentiment-analysis.streamlit.app/**](https://rbg-sentiment-analysis.streamlit.app/) 👈

**Fitur Dashboard:**
*   **Exploratory Data Analysis (EDA)**: Visualisasi data.
*   **LDA Visualization**: Visualisasi topik LDA interaktif.
*   **Model Evaluation**: Perbandingan metrik performa model.
*   **Sentiment Classifier Test**: Coba klasifikasikan ulasan Anda sendiri!

## 9. Pengaturan dan Penggunaan

### 9.1. Prasyarat

*   Python 3.9+
*   Pip atau Conda
*   Akun WandB (untuk logging jika menjalankan training)
*   API Key WandB (disimpan sebagai environment variable `WANDB_API_KEY` atau login via CLI)
*   (Opsional) GPU NVIDIA dengan CUDA untuk pelatihan yang dipercepat.

### 9.2. Instalasi

1.  **Clone Repositori:**
    ```bash
    git clone https://github.com/research-bocah-gundar/fesmaro-2025.git
    cd fesmaro-2025
    ```

2.  **Buat Virtual Environment (Direkomendasikan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Jika `requirements.txt` belum ada, Anda bisa membuatnya dari environment kerja Anda:
    ```bash
    pip freeze > requirements.txt
    ```
    Pastikan library untuk deployment (`streamlit`, `streamlit-option-menu`, `textstat`) juga termasuk jika Anda ingin menjalankan dashboard secara lokal.

4.  **Download Model SpaCy (jika belum terinstall oleh requirements):**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Siapkan Data**:
    *   Pastikan file data (`final_df.csv`, `df_val.csv`, `df_test.csv`, dll.) berada di direktori `data/`. Anda bisa mengunduhnya dari Kaggle atau sumber lain yang relevan.
    *   Jika menggunakan notebook `fesmaro-2025-analisis-big-data.ipynb`, jalankan untuk menghasilkan data yang diproses dan sampel.

### 9.3. Menjalankan Pelatihan

1.  **Konfigurasi**: Edit dictionary `config` di dalam script/notebook untuk memilih `model_type` yang diinginkan dan sesuaikan hyperparameter lain (batch size, learning rate, epochs, dll.).
2.  **Login WandB**: Pastikan Anda sudah login ke WandB (via API key atau `wandb login` di terminal).
3.  **Jalankan Pelatihan**: Eksekusi notebook atau script training.
    ```bash
    # Contoh jika menggunakan script python
    python src/train.py --model_type bert --epochs 5 --batch_size 64
    ```
    Proses pelatihan akan dimulai, metrik akan di-log ke WandB, dan model terbaik akan disimpan di `model/best_model.safetensors` (atau path yang ditentukan di config).

## 10. Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk detailnya. <!-- Buat file LICENSE jika belum ada -->
