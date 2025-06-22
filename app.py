import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import os
import json
import traceback

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen RS Al-Irsyad Surabaya",
    page_icon="🏥",
    layout="wide"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Download NLTK data jika belum ada
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')

download_nltk_data()

# Inisialisasi stemmer dan kamus singkatan
@st.cache_resource
def initialize_preprocessing():
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    kamus_singkatan = {
        'rs': 'rumah sakit', 'kpd': 'kepada', 'sby': 'surabaya', 'smg': 'semoga',
        'shg': 'sehingga', 'gak': 'tidak', 'tp': 'tapi', 'knp': 'kenapa',
        'kyk': 'kayak', 'yg': 'yang', 'klg': 'kalau', 'ttg': 'tentang',
        'jg': 'juga', 'rmh': 'rumah', 'th': 'tahun', 'lg': 'lagi',
        'dtg': 'datang', 'jd': 'jadi', 'bgt': 'banget', 'tdk': 'tidak',
        'gtw': 'tidak tahu', 'org': 'orang', 'byk': 'banyak', 'utk': 'untuk',
        'sy': 'saya', 'sya': 'saya', 'tgl': 'tanggal', 'dg': 'dengan',
        'feb': 'februari', 'kls': 'kelas', 'krn': 'karena', 'g': 'tidak',
        'gk': 'tidak', 'gue': 'saya', 'tj': 'tanjung', 'dn': 'dan',
        'wa': 'whatsapp', 'lt': 'lantai', 'dlm': 'dalam', 'skt': 'sakit',
        'd': 'di', 'blm': 'belum', 'ksh': 'kasih', 'bwt': 'buat',
        'km': 'kamu', 'sm': 'sama', 'dr': 'dari', 'mbk': 'mbak',
        'yt': 'youtube', 'gw': 'saya', 'dkt': 'dekat', 'sja': 'saja',
        'jgnkan': 'jangankan', 'ksih': 'kasih', 'gmna': 'gimana',
        'bsa': 'bisa', 'hrus': 'harus', 'nolak': 'tidak', 'smpk': 'sampai',
        'lol': 'jelek', 'kn': 'kena', 'krng': 'kurang', 'ad': 'ada',
        'dirs': 'di rumah sakit', 'sprt': 'seperti', 'klau': 'kalau',
        'krna': 'karena', 'dgn': 'dengan', 'udh': 'sudah', 'ga': 'tidak',
        'cma': 'cuma', 'kl': 'kalau', 'brp': 'berapa', 'sgt': 'sangat',
        'sdh': 'sudah', 'tlg': 'tolong', 'tpi': 'tapi', 'gak': 'tidak',
        'engga': 'tidak', 'enggak': 'tidak', 'nggak': 'tidak'
    }
    
    sentiment_words_to_keep = [
        'bagus', 'baik', 'cepat', 'ramah', 'bersih', 'nyaman', 'membantu', 'memuaskan',
        'buruk', 'lamban', 'parah', 'kotor', 'tidak', 'sopan', 'acuh', 'respon', 'murah',
        'mahal', 'puas', 'senang', 'marah', 'kecewa', 'tunggu', 'lama', 'profesional',
        'ramah', 'tidak ramah', 'tidak sopan', 'tidak memuaskan', 'senyum', 'menyenangkan',
        'jelek', 'kurang', 'bukan', 'lambat', 'kasar', 'galak', 'judes', 'jutek',
        'suster', 'perawat', 'dokter', 'staff', 'pelayanan', 'layanan',
        'kecil', 'jauh', 'sempit', 'penuh', 'susah', 'sulit', 'ribet', 'repot',
        'besar', 'dekat', 'luas', 'mudah', 'gampang', 'praktis', 'strategis'
    ]
    
    try:
        stopwords_id = stopwords.words('indonesian')
    except LookupError:
        nltk.download('stopwords')
        stopwords_id = stopwords.words('indonesian')
    
    return stemmer, kamus_singkatan, sentiment_words_to_keep, stopwords_id

stemmer, kamus_singkatan, sentiment_words_to_keep, stopwords_id = initialize_preprocessing()

# Fungsi preprocessing
def remove_numbers_punctuation_emoji(text):
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002500-\U00002BEF"
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    return text

def tokenize_text(text):
    if isinstance(text, str):
        return word_tokenize(text)
    else:
        return []

def normalisasi_singkatan(term):
    return kamus_singkatan.get(term, term)

def remove_stopwords(text, for_display=False):
    words = text.split()
    # Jangan hapus kata-kata penting untuk sentimen
    important_words = ['tidak', 'bukan', 'kurang', 'buruk', 'jelek', 'bagus', 'baik', 'ramah', 'tapi', 'tetapi', 'namun', 'judes', 'kasar', 'sopan']
    filtered_words = []
    for word in words:
        # Untuk display: jangan tampilkan kata dengan prefix NEG_ sama sekali
        if for_display and word.startswith('NEG_'):
            continue
        # Untuk model: tetap pakai NEG_ prefix, untuk display: skip NEG_ words
        elif not for_display and word.startswith('NEG_'):
            filtered_words.append(word)
        # Kata penting atau bukan stopword
        elif word.lower() in important_words or word.lower() not in stopwords_id:
            filtered_words.append(word)
    return " ".join(filtered_words)

def preprocess_text(text, term_dict, for_display=False):
    # Lowercase
    text = text.lower()
    
    # Handle negation patterns dan context switching
    negation_words = ['tidak', 'bukan', 'kurang', 'gak', 'ga', 'engga', 'enggak', 'nggak']
    context_reset_words = ['tapi', 'tetapi', 'namun', 'akan tetapi', 'meskipun', 'walaupun', 'sebaliknya']
    
    # Split berdasarkan context reset words untuk memisahkan klausa
    import re
    context_pattern = '|'.join(context_reset_words)
    clauses = re.split(f'({context_pattern})', text)
    
    processed_clauses = []
    
    for clause_idx, clause in enumerate(clauses):
        if clause.strip() in context_reset_words:
            processed_clauses.append(clause.strip())
            continue
            
        # Process each clause separately
        words = clause.split()
        processed_words = []
        negate_flag = False
        
        for i, word in enumerate(words):
            clean_word = remove_numbers_punctuation_emoji(word)
            
            if clean_word in negation_words:
                negate_flag = True
                processed_words.append(clean_word)
            elif negate_flag and clean_word.strip():
                # Untuk model: tambahkan prefix NEG_, untuk display: ganti dengan sinonim negatif
                if for_display:
                    # Untuk display: ganti dengan kata negatif yang natural
                    if clean_word in ['ramah', 'sopan']:
                        processed_words.append('kasar')
                    elif clean_word in ['baik', 'bagus']:
                        processed_words.append('buruk')
                    elif clean_word in ['bersih', 'nyaman']:
                        processed_words.append('kotor')
                    elif clean_word in ['cepat', 'mudah']:
                        processed_words.append('lambat')
                    elif clean_word in ['dekat', 'besar', 'luas']:
                        processed_words.append('kecil')
                    else:
                        processed_words.extend(['tidak', clean_word])
                else:
                    # Untuk model: gunakan NEG_ prefix
                    processed_words.append(f"NEG_{clean_word}")
                negate_flag = False
            else:
                processed_words.append(clean_word)
                # Reset negation hanya dalam klausa yang sama
                if word in [',', '.', 'dan', 'atau']:
                    negate_flag = False
        
        # Untuk klausa terakhir: berikan bobot ekstra MINIMAL hanya untuk model
        if clause_idx == len(clauses) - 1 and clause.strip() and not for_display:
            # Hanya duplikasi 1 kata sentiment terkuat di klausa terakhir (tidak semua)
            sentiment_found = False
            for word in processed_words:
                word_clean = word.lower().replace('neg_', '')
                if not sentiment_found and word_clean in ['judes', 'kasar', 'buruk', 'jelek', 'bagus', 'baik', 'ramah', 'sopan']:
                    # Duplikasi HANYA 1 kata sentiment pertama yang ditemukan
                    processed_words.append(word)
                    sentiment_found = True
                    break
        
        processed_clauses.append(' '.join(processed_words))
    
    # Join semua klausa kembali
    text = ' '.join(processed_clauses)
    
    # Hapus angka, tanda baca, dan emoji
    text = remove_numbers_punctuation_emoji(text)
    
    # Tokenisasi
    tokens = tokenize_text(text)
    
    # Normalisasi
    tokens_normalized = [normalisasi_singkatan(t) for t in tokens]
    
    # Stemming
    stemmed_tokens = []
    for term in tokens_normalized:
        if not for_display and term.startswith('NEG_'):
            # Untuk model: handle NEG_ prefix
            base_term = term[4:]  # Remove NEG_ prefix
            if base_term in sentiment_words_to_keep:
                stemmed_tokens.append(f"NEG_{base_term}")
            else:
                stemmed_tokens.append(f"NEG_{stemmer.stem(base_term)}")
        elif term in sentiment_words_to_keep:
            stemmed_tokens.append(term)
        else:
            stemmed_tokens.append(stemmer.stem(term))
    
    # Join dan hapus stopwords
    stemmed_text = ' '.join(stemmed_tokens)
    text = remove_stopwords(stemmed_text, for_display)
    
    return text

# Load model jika ada
def load_model():
    # Cek apakah file model sudah ada
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl') and os.path.exists('term_dict.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('term_dict.pkl', 'rb') as f:
            term_dict = pickle.load(f)
        return model, vectorizer, term_dict, None
    else:
        # Jika tidak ada, return None untuk semua
        return None, None, None, "Model belum dilatih. Silakan upload file data untuk melatih model."

# Fungsi untuk melatih model
def train_model(df):
    try:
        st.write("### 📋 Proses Training Model")
        
        # Step 1: Preprocessing
        st.write("**Step 1: Text Preprocessing**")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('Melakukan lowercase...')
        df['lower_case'] = df['text'].str.lower()
        progress_bar.progress(10)
        
        status_text.text('Menghapus angka, tanda baca, dan emoji...')
        df['remove_punctuation'] = df['lower_case'].apply(remove_numbers_punctuation_emoji)
        progress_bar.progress(20)
        
        status_text.text('Tokenisasi teks...')
        df['token'] = df['remove_punctuation'].apply(tokenize_text)
        progress_bar.progress(30)
        
        status_text.text('Normalisasi singkatan...')
        df['token_normalized'] = df['token'].apply(lambda doc: [normalisasi_singkatan(t) for t in doc])
        progress_bar.progress(40)
        
        # Build term dictionary untuk stemming
        status_text.text('Membangun dictionary untuk stemming...')
        term_dict = {}
        for document in df['token_normalized']:
            for term in document:
                if term not in term_dict:
                    if term in sentiment_words_to_keep:
                        term_dict[term] = term
                    else:
                        term_dict[term] = stemmer.stem(term)
        progress_bar.progress(50)
        
        def get_stemmed_term(document):
            return [term_dict[term] for term in document if term in term_dict]
        
        status_text.text('Melakukan stemming...')
        df['stemmed'] = df['token_normalized'].apply(lambda x: ' '.join(get_stemmed_term(x)))
        progress_bar.progress(60)
        
        status_text.text('Menghapus stopwords...')
        df['stopword_removal'] = df['stemmed'].apply(remove_stopwords)
        progress_bar.progress(70)
        
        # Tampilkan contoh preprocessing
        st.write("**Contoh Hasil Preprocessing:**")
        sample_idx = 0
        sample_data = {
            'Original': df['text'].iloc[sample_idx],
            'Lowercase': df['lower_case'].iloc[sample_idx],
            'Remove Punctuation': df['remove_punctuation'].iloc[sample_idx],
            'Tokens': ' | '.join(df['token'].iloc[sample_idx]),
            'Normalized': ' | '.join(df['token_normalized'].iloc[sample_idx]),
            'Stemmed': df['stemmed'].iloc[sample_idx],
            'Final': df['stopword_removal'].iloc[sample_idx]
        }
        
        for step, result in sample_data.items():
            st.text(f"{step}: {result}")
        
        # Step 2: Data Balancing
        st.write("\n**Step 2: Data Balancing**")
        
        # Tampilkan distribusi awal
        st.write("**Distribusi data sebelum balancing:**")
        original_counts = df['sentimen'].value_counts()
        for sentiment, count in original_counts.items():
            st.write(f"- {sentiment}: {count} samples")
        
        # Balance data dengan undersampling
        min_count = original_counts.min()
        st.write(f"\n**Melakukan balancing dengan undersampling ke {min_count} samples per kelas...**")
        
        balanced_dfs = []
        for sentiment in df['sentimen'].unique():
            sentiment_df = df[df['sentimen'] == sentiment].sample(n=min_count, random_state=42)
            balanced_dfs.append(sentiment_df)
        
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        # Tampilkan distribusi setelah balancing
        st.write("**Distribusi data setelah balancing:**")
        balanced_counts = df_balanced['sentimen'].value_counts()
        for sentiment, count in balanced_counts.items():
            st.write(f"- {sentiment}: {count} samples")
        
        progress_bar.progress(70)
        
        # Step 3: Split data
        st.write("\n**Step 3: Split Data Training dan Testing**")
        from sklearn.model_selection import train_test_split
        
        X = df_balanced['stopword_removal']
        y = df_balanced['sentimen']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        st.write(f"- Training data: {len(X_train)} samples")
        st.write(f"- Testing data: {len(X_test)} samples")
        st.write(f"- Training distribution: {pd.Series(y_train).value_counts().to_dict()}")
        st.write(f"- Testing distribution: {pd.Series(y_test).value_counts().to_dict()}")
        
        progress_bar.progress(75)
        
        # Step 4: TF-IDF Vectorization
        status_text.text('Melakukan TF-IDF Vectorization...')
        st.write("\n**Step 4: TF-IDF Vectorization**")
        # Tingkatkan ngram_range untuk menangkap konteks yang lebih panjang
        # TF-IDF Vectorization sederhana
        vectorizer = TfidfVectorizer(
            max_features=8000, 
            ngram_range=(1, 3),  # Unigram, bigram, dan trigram
            sublinear_tf=True,
            min_df=1,  # Minimal muncul di 1 dokumen
            max_df=0.95  # Maksimal muncul di 95% dokumen
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        st.write(f"- Feature matrix shape: {X_train_tfidf.shape}")
        st.write(f"- Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        progress_bar.progress(85)
        
        # Step 5: Train model
        status_text.text('Melatih model Naive Bayes...')
        st.write("\n**Step 5: Training Naive Bayes Model**")
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)
        
        progress_bar.progress(95)
        
        # Step 6: Evaluasi model
        status_text.text('Mengevaluasi model...')
        st.write("\n**Step 6: Evaluasi Model**")
        
        # Prediksi
        y_pred = model.predict(X_test_tfidf)
        
        # Akurasi
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Akurasi Model: {accuracy:.4f} ({accuracy*100:.2f}%)**")
        
        # Classification Report
        st.write("**Classification Report:**")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4))
        
        # Confusion Matrix
        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negatif', 'Positif'], 
                   yticklabels=['Negatif', 'Positif'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        progress_bar.progress(100)
        status_text.text('Training selesai!')
        
        # Test model dengan contoh sederhana SEBELUM save
        st.write("\n**Step 7: Testing Model dengan Konteks Sentiment Akhir**")
        test_cases = [
            ("dokter ramah sekali", "positif"),
            ("dokter tidak ramah", "negatif"), 
            ("pelayanan bagus", "positif"),
            ("pelayanan buruk", "negatif"),
            ("rumah sakit bersih", "positif"),
            ("rumah sakit kotor", "negatif"),
            ("pelayanan jelek", "negatif"),
            ("satpam tidak sopan", "negatif"),
            ("rumah sakit bagus tapi pelayanan jelek", "negatif"),  # Akhiran negatif
            ("pelayanan bagus tapi dokter tidak ramah", "negatif"),  # Akhiran negatif
            ("fasilitas buruk tapi pelayanan memuaskan", "positif"),  # Akhiran positif
            ("awalnya bagus tapi akhirnya mengecewakan", "negatif"),  # Akhiran negatif
            ("pelayanan buruk tapi dokter baik", "positif"),  # Akhiran positif
            ("dokter baik namun perawat kasar", "negatif"),  # Akhiran negatif
            ("rumah sakit nya bagus tapi suster nya judes", "negatif"),  # Kasus user
            ("fasilitas bersih tapi suster galak", "negatif")  # Variasi kasus user
        ]
        
        st.write("**Debug: Melihat proses preprocessing dengan context-aware sentiment**")
        
        correct_predictions = 0
        for test_text, expected in test_cases:
            st.write(f"\n🔍 **Debugging: '{test_text}' (Expected: {expected})**")
            
            # Step 1: Lowercase
            step1 = test_text.lower()
            st.write(f"1. Lowercase: `{step1}`")
            
            # Step 2: Context-aware processing
            negation_words = ['tidak', 'bukan', 'kurang', 'gak', 'ga', 'engga', 'enggak', 'nggak']
            context_reset_words = ['tapi', 'tetapi', 'namun', 'akan tetapi', 'meskipun', 'walaupun', 'sebaliknya']
            
            # Split by context reset words
            import re
            context_pattern = '|'.join(context_reset_words)
            clauses = re.split(f'({context_pattern})', step1)
            
            st.write("2. Context-aware clause splitting:")
            st.write(f"   - Clauses: {clauses}")
            
            processed_clauses = []
            for clause_idx, clause in enumerate(clauses):
                if clause.strip() in context_reset_words:
                    processed_clauses.append(clause.strip())
                    st.write(f"   - Context reset word: '{clause.strip()}'")
                    continue
                    
                words = clause.split()
                processed_words = []
                negate_flag = False
                
                for word in words:
                    clean_word = remove_numbers_punctuation_emoji(word)
                    if clean_word in negation_words:
                        negate_flag = True
                        processed_words.append(clean_word)
                    elif negate_flag and clean_word.strip():
                        # Untuk debug display: tampilkan natural
                        if clean_word in ['ramah', 'sopan']:
                            processed_words.extend(['kasar', 'judes'])
                        elif clean_word in ['baik', 'bagus']:
                            processed_words.extend(['buruk', 'jelek'])
                        elif clean_word in ['bersih', 'nyaman']:
                            processed_words.extend(['kotor'])
                        else:
                            processed_words.extend(['tidak', clean_word])
                        negate_flag = False
                    else:
                        processed_words.append(clean_word)
                
                # Debugging: tidak perlu duplikasi untuk bobot, cukup tampilkan apa adanya
                processed_clauses.append(' '.join(processed_words))
            
            step2 = ' '.join(processed_clauses)
            st.write(f"   Result: `{step2}`")
            
            # Step 3: Remove punctuation  
            step3 = remove_numbers_punctuation_emoji(step2)
            st.write(f"3. Remove punct: `{step3}`")
            
            # Step 4: Tokenize
            step4 = tokenize_text(step3)
            st.write(f"4. Tokens: `{step4}`")
            
            # Step 5: Normalize
            step5 = [normalisasi_singkatan(t) for t in step4]
            st.write(f"5. Normalized: `{step5}`")
            
            # Step 6: Stemming sederhana (untuk display natural)
            step6 = []
            for term in step5:
                if term in sentiment_words_to_keep:
                    step6.append(term)
                    st.write(f"   - KEPT: '{term}' (sentiment word)")
                else:
                    stemmed = stemmer.stem(term)
                    step6.append(stemmed)
                    st.write(f"   - STEMMED: '{term}' → '{stemmed}'")
                    st.write(f"   - STEMMED: '{term}' → '{stemmed}'")
            
            step6_text = ' '.join(step6)
            st.write(f"6. Stemmed: `{step6_text}`")
            
            # Step 7: Remove stopwords (untuk display natural)
            step7 = remove_stopwords(step6_text, for_display=True)
            st.write(f"7. Final (after stopwords): `{step7}`")
            
            if step7.strip():
                # Vectorize
                test_tfidf = vectorizer.transform([step7])
                
                # Check which features are detected
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = test_tfidf.toarray()[0]
                active_features = [(feature_names[i], tfidf_scores[i]) for i in range(len(tfidf_scores)) if tfidf_scores[i] > 0]
                
                st.write(f"7. TF-IDF features detected: `{active_features}`")
                
                # Predict
                prediction = model.predict(test_tfidf)[0]
                probabilities = model.predict_proba(test_tfidf)[0]
                
                # Get class labels
                classes = model.classes_
                
                # Check correctness
                is_correct = prediction == expected
                if is_correct:
                    correct_predictions += 1
                    status_icon = "✅"
                else:
                    status_icon = "❌"
                
                # Display probabilities for each class
                prob_dict = dict(zip(classes, probabilities))
                
                st.write(f"8. **RESULT: {status_icon} Prediksi: {prediction} | Probabilitas: {prob_dict}**")
            else:
                st.write("❌ **EMPTY TEXT after preprocessing!**")
            
            st.write("---")
        
        test_accuracy = correct_predictions / len(test_cases)
        st.write(f"**Test Accuracy pada contoh sederhana: {test_accuracy:.2%} ({correct_predictions}/{len(test_cases)})**")
        
        if test_accuracy < 0.6:
            st.error("🚨 Model perlu perbaikan untuk menangani mixed sentiment!")
            st.write("**Model belum optimal dalam menangani:**")
            st.write("- Context switching dengan 'tapi', 'tetapi', 'namun'")
            st.write("- Prioritas sentiment di akhir kalimat")  
            st.write("- Kombinasi negasi dan context switching")
        else:
            st.success("✅ Model berhasil menangani mixed sentiment dengan baik!")
        
        # Save model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open('term_dict.pkl', 'wb') as f:
            pickle.dump(term_dict, f)
        
        # Save preprocessed data dengan semua kolom (data yang sudah di-balance)
        df_balanced['preprocessing_text'] = df_balanced['stopword_removal']
        df_balanced.to_csv('preprocessing_text.csv', index=False)
        
        # Save original data (tidak di-balance) untuk keperluan analisis
        df['preprocessing_text'] = df['stopword_removal']
        df.to_csv('preprocessing_text_full.csv', index=False)
        
        # Save evaluation results
        eval_results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_count': len(vectorizer.vocabulary_),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'original_data_size': len(df),
            'balanced_data_size': len(df_balanced),
            'original_distribution': original_counts.to_dict(),
            'balanced_distribution': balanced_counts.to_dict()
        }
        
        import json
        with open('evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return model, vectorizer, term_dict, df_balanced
        
    except Exception as e:
        st.error(f"❌ Error saat melatih model: {str(e)}")
        import traceback
        st.error(f"Detail error: {traceback.format_exc()}")
        return None, None, None, None

# Inisialisasi session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'term_dict' not in st.session_state:
    st.session_state.term_dict = None

# Load model pada startup
if not st.session_state.model_trained:
    model, vectorizer, term_dict, error_message = load_model()
    if model is not None:
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.term_dict = term_dict
        st.session_state.model_trained = True
        st.session_state.error_message = None
    else:
        st.session_state.error_message = error_message
else:
    model = st.session_state.model
    vectorizer = st.session_state.vectorizer
    term_dict = st.session_state.term_dict
    error_message = st.session_state.get('error_message', None)

# Header
st.markdown('<h1 class="main-header">🏥 Analisis Sentimen Ulasan RS Al-Irsyad Surabaya</h1>', unsafe_allow_html=True)

# Status Model
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.session_state.model_trained:
        st.success("✅ Model sudah dilatih dan siap digunakan!")
    else:
        st.warning("⚠️ Model belum dilatih. Silakan latih model di menu Beranda.")

st.markdown("---")

# Sidebar
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["🏠 Beranda", "📊 Analisis Data", "🔮 Prediksi Sentimen", "📈 Visualisasi"]
)

if menu == "🏠 Beranda":
    st.markdown('<h2 class="section-header">Selamat Datang!</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        ### Tentang Aplikasi
        
        Aplikasi ini menggunakan metode **Naive Bayes** untuk menganalisis sentimen ulasan 
        Rumah Sakit Al-Irsyad Surabaya. Aplikasi dapat:
        
        - 📊 Menganalisis distribusi sentimen dari data yang ada
        - 🔍 Melakukan preprocessing teks otomatis
        - 🤖 Memprediksi sentimen ulasan baru
        - 📈 Menampilkan visualisasi data yang menarik
        
        ### Fitur Preprocessing:
        - Normalisasi teks (lowercase, hapus tanda baca)
        - Tokenisasi dan normalisasi singkatan
        - Stemming dengan Sastrawi
        - Stopword removal
        - TF-IDF Vectorization
        """)
    
    with col2:
        st.info("""
        ### 📋 Cara Penggunaan
        
        1. **Upload Data**: Upload file CSV dengan kolom 'text' dan 'sentimen'
        2. **Analisis**: Lihat distribusi dan statistik data
        3. **Prediksi**: Masukkan ulasan baru untuk diprediksi
        4. **Visualisasi**: Lihat word cloud dan grafik
        """)
    
    # Data dan Model section
    st.markdown('<h2 class="section-header">Data dan Model Training</h2>', unsafe_allow_html=True)
    
    # Auto-load data dari data.csv
    csv_file = "data.csv"
    if os.path.exists(csv_file):
        st.success(f"✅ Data tersedia: {csv_file}")
        
        # Load and preview data
        try:
            df = pd.read_csv(csv_file)
            # Rename snippet to text untuk konsistensi jika diperlukan
            if 'snippet' in df.columns:
                df = df.rename(columns={'snippet': 'text'})
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Preview Data:**")
                st.dataframe(df[['text', 'sentimen']].head())
                st.write(f"Total data: {len(df)} baris")
                
                # Tampilkan distribusi sentimen
                sentiment_counts = df['sentimen'].value_counts()
                st.write("**Distribusi Sentimen:**")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(df)) * 100
                    st.write(f"- {sentiment}: {count} ({percentage:.1f}%)")
            
            with col2:
                st.info("""
                **� Data Statistik**
                
                Data ini berisi ulasan rumah sakit dengan label sentimen yang sudah diklasifikasi.
                """)
                
                # Train model button - lebih prominent
                if st.button("🚀 Latih Model Baru", key="train_model", help="Klik untuk melatih ulang model dengan data terbaru"):
                    with st.spinner('Sedang melatih model...'):
                        # Clear old model files first
                        for old_file in ['model.pkl', 'vectorizer.pkl', 'term_dict.pkl']:
                            if os.path.exists(old_file):
                                os.remove(old_file)
                        
                        result = train_model(df)
                        if result[0] is not None:  # Jika training berhasil
                            # Update session state dengan model baru
                            st.session_state.model = result[0]
                            st.session_state.vectorizer = result[1]
                            st.session_state.term_dict = result[2]
                            st.session_state.model_trained = True
                            st.session_state.error_message = None
                            
                            st.balloons()
                            st.success("🎉 Model berhasil dilatih dengan preprocessing yang diperbaiki!")
                            st.info("Model sekarang dapat menangani mixed sentiment dengan lebih baik!")
                            
                            # Test quick prediction to verify
                            st.write("**Quick Test:**")
                            test_text = "pelayanan buruk sekali"
                            processed = preprocess_text(test_text, result[2], for_display=False)
                            if processed.strip():
                                test_tfidf = result[1].transform([processed])
                                pred = result[0].predict(test_tfidf)[0]
                                prob = result[0].predict_proba(test_tfidf)[0]
                                st.write(f"Test: '{test_text}' → {pred} (confidence: {max(prob):.2%})")
                        else:
                            st.error("❌ Gagal melatih model. Silakan coba lagi.")
                
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")
    else:
        st.error(f"❌ File {csv_file} tidak ditemukan!")
        st.info("Pastikan file data.csv ada di direktori yang sama dengan app.py")

elif menu == "📊 Analisis Data":
    st.markdown('<h2 class="section-header">Analisis Data</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("Model belum dilatih. Silakan upload data di menu Beranda untuk melatih model.")
        st.info("Silakan upload data di menu Beranda untuk melatih model.")
    else:
        # Load sample data untuk analisis
        try:
            # Tampilkan hasil evaluasi model jika ada
            if os.path.exists('evaluation_results.json'):
                st.subheader("🎯 Evaluasi Model")
                
                import json
                with open('evaluation_results.json', 'r') as f:
                    eval_results = json.load(f)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Akurasi", f"{eval_results['accuracy']:.3f}")
                with col2:
                    st.metric("Total Fitur", eval_results['feature_count'])
                with col3:
                    st.metric("Data Training", eval_results['train_size'])
                with col4:
                    st.metric("Data Testing", eval_results['test_size'])
                
                # Tampilkan informasi balancing jika ada
                if 'original_data_size' in eval_results:
                    st.write("**Informasi Data Balancing:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Data Original:**")
                        st.write(f"- Total: {eval_results['original_data_size']} samples")
                        for sentiment, count in eval_results['original_distribution'].items():
                            st.write(f"- {sentiment}: {count} samples")
                    
                    with col2:
                        st.write("**Data Setelah Balancing:**")
                        st.write(f"- Total: {eval_results['balanced_data_size']} samples")
                        for sentiment, count in eval_results['balanced_distribution'].items():
                            st.write(f"- {sentiment}: {count} samples")
                
                # Classification Report
                st.write("**Classification Report:**")
                report_df = pd.DataFrame(eval_results['classification_report']).transpose()
                st.dataframe(report_df.round(4))
                
                # Confusion Matrix
                st.write("**Confusion Matrix:**")
                cm = np.array(eval_results['confusion_matrix'])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Negatif', 'Positif'], 
                           yticklabels=['Negatif', 'Positif'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            
            # Coba load data yang sudah diproses
            if os.path.exists('preprocessing_text.csv'):
                df = pd.read_csv('preprocessing_text.csv')
                
                st.subheader("📊 Statistik Data")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Data", len(df))
                
                with col2:
                    positive_count = len(df[df['sentimen'] == 'positif'])
                    st.metric("Sentimen Positif", positive_count)
                
                with col3:
                    negative_count = len(df[df['sentimen'] == 'negatif'])
                    st.metric("Sentimen Negatif", negative_count)
                
                # Distribusi sentimen
                st.subheader("📊 Distribusi Sentimen")
                fig, ax = plt.subplots(figsize=(8, 6))
                sentiment_counts = df['sentimen'].value_counts()
                colors = ['#28a745', '#dc3545']
                ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
                ax.set_title('Distribusi Sentimen')
                st.pyplot(fig)
                
                # Statistik panjang teks
                st.subheader("📏 Statistik Panjang Teks")
                df['text_length'] = df['preprocessing_text'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Statistik Deskriptif:**")
                    st.write(df['text_length'].describe())
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(df['text_length'], bins=30, alpha=0.7, color='skyblue')
                    ax.set_xlabel('Panjang Teks (jumlah kata)')
                    ax.set_ylabel('Frekuensi')
                    ax.set_title('Distribusi Panjang Teks')
                    st.pyplot(fig)
                
                # Tampilkan contoh preprocessing
                st.subheader("🔍 Contoh Proses Preprocessing")
                sample_options = st.selectbox("Pilih contoh data:", range(min(10, len(df))))
                
                if sample_options is not None:
                    sample_row = df.iloc[sample_options]
                    
                    preprocessing_steps = {
                        'Original Text': sample_row.get('text', 'N/A'),
                        'Lowercase': sample_row.get('lower_case', 'N/A'), 
                        'Remove Punctuation': sample_row.get('remove_punctuation', 'N/A'),
                        'Tokens': sample_row.get('token', 'N/A'),
                        'Normalized': sample_row.get('token_normalized', 'N/A'),
                        'Stemmed': sample_row.get('stemmed', 'N/A'),
                        'Final (No Stopwords)': sample_row.get('preprocessing_text', 'N/A'),
                        'Sentiment': sample_row.get('sentimen', 'N/A')
                    }
                    
                    for step, text in preprocessing_steps.items():
                        if step == 'Sentiment':
                            sentiment_color = '🟢' if text == 'positif' else '🔴'
                            st.write(f"**{step}:** {sentiment_color} {text}")
                        else:
                            st.write(f"**{step}:** {text}")
                
            else:
                st.info("Data belum tersedia. Silakan upload data terlebih dahulu.")
                
        except Exception as e:
            st.error(f"Error dalam analisis data: {str(e)}")
            import traceback
            st.error(f"Detail error: {traceback.format_exc()}")

elif menu == "🔮 Prediksi Sentimen":
    st.markdown('<h2 class="section-header">Prediksi Sentimen Ulasan</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("Model belum dilatih. Silakan upload data di menu Beranda untuk melatih model.")
        st.info("Silakan upload data di menu Beranda untuk melatih model.")
    else:
        st.write("Masukkan ulasan untuk diprediksi sentimennya:")
        
        # Input methods
        input_method = st.radio("Pilih metode input:", ["Single Input", "Multiple Input"])
        
        if input_method == "Single Input":
            user_input = st.text_area(
                "Masukkan ulasan:",
                placeholder="Contoh: Pelayanan rumah sakit ini sangat cepat dan memuaskan",
                height=100
            )
            
            if st.button("🔍 Prediksi Sentimen"):
                if user_input.strip():
                    try:
                        st.write("### 🔍 Proses Prediksi:")
                        
                        # Tampilkan teks original
                        st.write("**1. Teks Original:**")
                        st.code(user_input)
                        
                        # Preprocess step by step
                        st.write("**2. Proses Preprocessing:**")
                        
                        # Lowercase
                        text_lower = user_input.lower()
                        st.write(f"- Lowercase: `{text_lower}`")
                        
                        # Remove punctuation
                        text_clean = remove_numbers_punctuation_emoji(text_lower)
                        st.write(f"- Remove punctuation: `{text_clean}`")
                        
                        # Tokenize
                        tokens = tokenize_text(text_clean)
                        st.write(f"- Tokens: `{' | '.join(tokens)}`")
                        
                        # Normalize
                        tokens_normalized = [normalisasi_singkatan(t) for t in tokens]
                        st.write(f"- Normalized: `{' | '.join(tokens_normalized)}`")
                        
                        # Stemming
                        stemmed_tokens = []
                        for term in tokens_normalized:
                            if term in sentiment_words_to_keep:
                                stemmed_tokens.append(term)
                            else:
                                stemmed_tokens.append(stemmer.stem(term))
                        
                        stemmed_text = ' '.join(stemmed_tokens)
                        st.write(f"- Stemmed: `{stemmed_text}`")
                        
                        # Remove stopwords
                        final_text = remove_stopwords(stemmed_text)
                        st.write(f"- Final (after stopword removal): `{final_text}`")
                        
                        if not final_text.strip():
                            st.warning("⚠️ Teks kosong setelah preprocessing. Coba masukkan teks yang lebih panjang.")
                            st.stop()
                        
                        st.write("**3. Vectorization dan Prediksi:**")
                        
                        # Vectorize
                        text_tfidf = st.session_state.vectorizer.transform([final_text])
                        st.write(f"- TF-IDF vector shape: {text_tfidf.shape}")
                        st.write(f"- Non-zero features: {text_tfidf.nnz}")
                        
                        # Show top features
                        if text_tfidf.nnz > 0:
                            feature_names = st.session_state.vectorizer.get_feature_names_out()
                            tfidf_scores = text_tfidf.toarray()[0]
                            top_features = [(feature_names[i], tfidf_scores[i]) for i in tfidf_scores.argsort()[-10:][::-1] if tfidf_scores[i] > 0]
                            
                            if top_features:
                                st.write("- Top features detected:")
                                for feature, score in top_features:
                                    st.write(f"  - `{feature}`: {score:.4f}")
                        
                        # Predict
                        prediction = st.session_state.model.predict(text_tfidf)[0]
                        probabilities = st.session_state.model.predict_proba(text_tfidf)[0]
                        
                        # Get class labels
                        classes = st.session_state.model.classes_
                        
                        st.write("**4. Hasil Prediksi:**")
                        for i, class_label in enumerate(classes):
                            st.write(f"- Probabilitas {class_label}: {probabilities[i]:.4f} ({probabilities[i]*100:.2f}%)")
                        
                        # Display final result
                        st.markdown("### 🎯 Hasil Akhir:")
                        
                        if prediction == 'positif':
                            pos_idx = list(classes).index('positif')
                            conf_score = probabilities[pos_idx]
                            st.markdown(f"""
                            <div class="prediction-box positive">
                                <h4>😊 Sentimen: POSITIF</h4>
                                <p><strong>Confidence:</strong> {conf_score:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            neg_idx = list(classes).index('negatif')
                            conf_score = probabilities[neg_idx]
                            st.markdown(f"""
                            <div class="prediction-box negative">
                                <h4>😞 Sentimen: NEGATIF</h4>
                                <p><strong>Confidence:</strong> {conf_score:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Error dalam prediksi: {str(e)}")
                        import traceback
                        st.error(f"Detail error: {traceback.format_exc()}")
                else:
                    st.warning("Silakan masukkan ulasan terlebih dahulu.")
        
        else:  # Multiple Input
            st.write("Masukkan beberapa ulasan (satu ulasan per baris):")
            user_inputs = st.text_area(
                "Ulasan (pisahkan dengan enter):",
                placeholder="Pelayanan rumah sakit ini sangat cepat\nParkirnya luas tapi satpam kurang baik\nFasilitasnya bersih dan nyaman",
                height=150
            )
            
            # Tambahkan test cases
            st.write("**Atau gunakan contoh test cases:**")
            if st.button("🧪 Test dengan Contoh Kasus"):
                test_cases = [
                    "Pelayanan sangat memuaskan dan dokter ramah sekali",
                    "Pelayanan buruk sekali, dokter tidak ramah",
                    "Rumah sakit bersih dan nyaman",
                    "Tempat kotor dan tidak terawat",
                    "Staff sangat membantu dan profesional", 
                    "Staff acuh tak acuh dan tidak sopan",
                    "Rumah sakit bagus tapi pelayanan jelek",  # Test kasus yang Anda sebutkan
                    "Fasilitas bersih tetapi dokter tidak ramah",
                    "Awalnya bagus tapi akhirnya mengecewakan",
                    "Dokter baik namun perawat kurang sopan"
                ]
                
                st.write("**Hasil Test Cases:**")
                for i, case in enumerate(test_cases, 1):
                    # Untuk model: gunakan preprocessing lengkap dengan NEG_ prefix
                    processed_text = preprocess_text(case, st.session_state.term_dict, for_display=False)
                    # Untuk display: gunakan preprocessing natural tanpa NEG_ prefix
                    display_text = preprocess_text(case, st.session_state.term_dict, for_display=True)
                    
                    text_tfidf = st.session_state.vectorizer.transform([processed_text])
                    prediction = st.session_state.model.predict(text_tfidf)[0]
                    probability = st.session_state.model.predict_proba(text_tfidf)[0]
                    classes = st.session_state.model.classes_
                    
                    if prediction == 'positif':
                        pos_idx = list(classes).index('positif')
                        confidence = probability[pos_idx]
                        emoji = "😊"
                        color = "🟢"
                    else:
                        neg_idx = list(classes).index('negatif')
                        confidence = probability[neg_idx]
                        emoji = "😞"
                        color = "🔴"
                    
                    st.write(f"{i}. **{case}**")
                    st.write(f"   {color} {emoji} **{prediction.upper()}** ({confidence:.2%}) | Processed: `{display_text}`")
                    st.write("")
            
            if st.button("🔍 Prediksi Semua Sentimen"):
                if user_inputs.strip():
                    try:
                        reviews = [review.strip() for review in user_inputs.split('\n') if review.strip()]
                        
                        if reviews:
                            results = []
                            
                            for i, review in enumerate(reviews):
                                # Preprocess untuk model: dengan NEG_ prefix
                                processed_text = preprocess_text(review, st.session_state.term_dict, for_display=False)
                                # Preprocess untuk display: natural tanpa NEG_ prefix  
                                display_text = preprocess_text(review, st.session_state.term_dict, for_display=True)
                                
                                # Vectorize
                                text_tfidf = st.session_state.vectorizer.transform([processed_text])
                                
                                # Predict
                                prediction = st.session_state.model.predict(text_tfidf)[0]
                                probability = st.session_state.model.predict_proba(text_tfidf)[0]
                                classes = st.session_state.model.classes_
                                
                                # Get correct confidence score
                                if prediction == 'positif':
                                    pos_idx = list(classes).index('positif')
                                    confidence = probability[pos_idx]
                                else:
                                    neg_idx = list(classes).index('negatif')
                                    confidence = probability[neg_idx]
                                
                                results.append({
                                    'No': i + 1,
                                    'Ulasan': review,
                                    'Processed': display_text,  # Gunakan display_text yang natural
                                    'Sentimen': prediction,
                                    'Confidence': f"{confidence:.2%}"
                                })
                            
                            # Display results
                            st.markdown("### 🎯 Hasil Prediksi:")
                            results_df = pd.DataFrame(results)
                            
                            # Style the dataframe
                            def highlight_sentiment(val):
                                if val == 'positif':
                                    return 'background-color: #d4edda'
                                elif val == 'negatif':
                                    return 'background-color: #f8d7da'
                                return ''
                            
                            styled_df = results_df.style.applymap(highlight_sentiment, subset=['Sentimen'])
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Summary
                            positive_count = len([r for r in results if r['Sentimen'] == 'positif'])
                            negative_count = len([r for r in results if r['Sentimen'] == 'negatif'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Ulasan", len(results))
                            with col2:
                                st.metric("Positif", positive_count)
                            with col3:
                                st.metric("Negatif", negative_count)
                        
                    except Exception as e:
                        st.error(f"Error dalam prediksi: {str(e)}")
                else:
                    st.warning("Silakan masukkan ulasan terlebih dahulu.")

elif menu == "📈 Visualisasi":
    st.markdown('<h2 class="section-header">Visualisasi Data</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("Model belum dilatih. Silakan upload data di menu Beranda untuk melatih model.")
        st.info("Silakan upload data di menu Beranda untuk melatih model.")
    else:
        try:
            if os.path.exists('preprocessing_text.csv'):
                df = pd.read_csv('preprocessing_text.csv')
                
                # Word Cloud
                st.subheader("☁️ Word Cloud")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Word Cloud - Sentimen Positif**")
                    positive_text = " ".join(df[df['sentimen'] == 'positif']['preprocessing_text'])
                    if positive_text.strip():
                        wordcloud_pos = WordCloud(
                            width=400, height=300, 
                            background_color='white', 
                            colormap='Greens'
                        ).generate(positive_text)
                        
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.imshow(wordcloud_pos, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.info("Tidak ada data sentimen positif")
                
                with col2:
                    st.write("**Word Cloud - Sentimen Negatif**")
                    negative_text = " ".join(df[df['sentimen'] == 'negatif']['preprocessing_text'])
                    if negative_text.strip():
                        wordcloud_neg = WordCloud(
                            width=400, height=300, 
                            background_color='white', 
                            colormap='Reds'
                        ).generate(negative_text)
                        
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.imshow(wordcloud_neg, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.info("Tidak ada data sentimen negatif")
                
                # Top words
                st.subheader("🔤 Kata-kata Paling Sering Muncul")
                
                from collections import Counter
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top 10 Kata - Sentimen Positif**")
                    pos_words = " ".join(df[df['sentimen'] == 'positif']['preprocessing_text']).split()
                    pos_counter = Counter(pos_words)
                    pos_top = pos_counter.most_common(10)
                    
                    if pos_top:
                        pos_df = pd.DataFrame(pos_top, columns=['Kata', 'Frekuensi'])
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.barh(pos_df['Kata'], pos_df['Frekuensi'], color='green', alpha=0.7)
                        ax.set_xlabel('Frekuensi')
                        ax.set_title('Top 10 Kata - Sentimen Positif')
                        plt.tight_layout()
                        st.pyplot(fig)
                
                with col2:
                    st.write("**Top 10 Kata - Sentimen Negatif**")
                    neg_words = " ".join(df[df['sentimen'] == 'negatif']['preprocessing_text']).split()
                    neg_counter = Counter(neg_words)
                    neg_top = neg_counter.most_common(10)
                    
                    if neg_top:
                        neg_df = pd.DataFrame(neg_top, columns=['Kata', 'Frekuensi'])
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.barh(neg_df['Kata'], neg_df['Frekuensi'], color='red', alpha=0.7)
                        ax.set_xlabel('Frekuensi')
                        ax.set_title('Top 10 Kata - Sentimen Negatif')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
            else:
                st.info("Data belum tersedia untuk visualisasi.")
                
        except Exception as e:
            st.error(f"Error dalam membuat visualisasi: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    <p>🏥 Analisis Sentimen RS Al-Irsyad Surabaya | Menggunakan Naive Bayes & Streamlit</p>
</div>
""", unsafe_allow_html=True)
