# Sentiment Analysis Aplikasi Ulasan Rumah Sakit

Aplikasi analisis sentimen berbasis Streamlit untuk menganalisis ulasan Rumah Sakit Al-Irsyad Surabaya menggunakan metode Naive Bayes.

## 🚀 Demo

Aplikasi ini dapat dijalankan secara langsung dengan menganalisis sentimen dari teks ulasan rumah sakit.

## ✨ Fitur

- **Preprocessing Text Otomatis**: Pembersihan teks, stemming, dan penghapusan stopwords
- **Negation Handling**: Menangani kata negasi dalam konteks sentimen
- **Context Switching**: Menangani kata transisi seperti "tapi", "namun"
- **Mixed Sentiment Detection**: Deteksi sentimen campuran dengan prioritas konteks akhir
- **Visualisasi**: WordCloud dan distribusi sentimen
- **Real-time Prediction**: Prediksi sentimen secara real-time

## 🛠️ Teknologi

- **Streamlit**: Framework web app
- **Naive Bayes**: Algoritma machine learning
- **TF-IDF**: Feature extraction
- **Sastrawi**: Stemming Bahasa Indonesia
- **NLTK**: Natural Language Processing

## 📁 File Structure

```
├── app.py                 # Main application
├── data.csv              # Dataset training
├── model.pkl             # Trained model
├── vectorizer.pkl        # TF-IDF vectorizer
├── term_dict.pkl         # Term dictionary
├── requirements.txt      # Dependencies
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## 🚀 Deployment

### Local Deployment

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud

1. Push code ke GitHub repository
2. Connect ke Streamlit Cloud
3. Deploy otomatis dari repository

## 📊 Model Performance

Model telah dilatih dan dievaluasi dengan hasil yang tersimpan dalam `evaluation_results.json`.

## 🔧 Cara Penggunaan

1. Buka aplikasi
2. Masukkan teks ulasan rumah sakit
3. Klik "Analisis Sentimen"
4. Lihat hasil prediksi dan visualisasi

## 📝 Contoh Input

- "Pelayanan rumah sakit sangat baik dan dokternya ramah"
- "Rumah sakit bagus tapi suster judes"
- "Fasilitas kurang lengkap dan pelayanan lambat"

---

Dibuat untuk analisis sentimen ulasan Rumah Sakit Al-Irsyad Surabaya
