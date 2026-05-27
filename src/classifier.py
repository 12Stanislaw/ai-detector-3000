import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Імпортуємо ваші лінгвістичні метрики
from src.metrics_basic import calculate_avg_sentence_length, calculate_sentence_length_variance
from src.metrics_advanced import calculate_ttr, calculate_adjective_density

_model = None
_vectorizer = None
_scaler = None

def clean_text_basic(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Тільки англійські літери
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_linguistic_features(texts_series):
    """
    Приймає колонку з текстами та повертає звичайний матричний масив 
    із 4 лінгвістичних ознак (метрики Стаса та Богдана).
    """
    features_list = []
    for text in texts_series:
        if not isinstance(text, str):
            text = ""
        avg_sent_len = calculate_avg_sentence_length(text)
        sent_len_var = calculate_sentence_length_variance(text)
        ttr = calculate_ttr(text)
        adj_density = calculate_adjective_density(text)
        
        features_list.append([avg_sent_len, sent_len_var, ttr, adj_density])
    return np.array(features_list)

def train_detector(dataset_path="dataset.csv"):
    global _model, _vectorizer, _scaler
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset path {dataset_path} not found!")

    print("⏳ Завантаження датасету для комбінованого навчання...")
    df = pd.read_csv(dataset_path, encoding='utf-8')
    
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    df['cleaned_text'] = df[text_col].apply(clean_text_basic)
    
    X_raw = df['cleaned_text']
    X_orig = df[text_col]  # Для лінгвістики потрібен оригінал із розділовими знаками
    y = df['generated']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    X_train_orig, X_test_orig, _, _ = train_test_split(X_orig, y, test_size=0.2, random_state=42)

    print("📊 Рахуємо TF-IDF (обмежено до 500 слів)...")
    _vectorizer = TfidfVectorizer(max_features=500, min_df=2, stop_words='english')
    X_train_tfidf = _vectorizer.fit_transform(X_train_raw)
    
    print("📈 Рахуємо лінгвістичні метрики...")
    X_train_ling = extract_linguistic_features(X_train_orig)
    
    print("⚖️ Масштабуємо лінгвістичні метрики за допомогою StandardScaler...")
    _scaler = StandardScaler()
    X_train_ling_scaled = _scaler.fit_transform(X_train_ling)
    
    print("🔗 Зшиваємо ознаки разом (TF-IDF + Scaled Linguistics)...")
    X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_ling_scaled)])
    
    print("🤖 Навчаємо фінальну модель (з регуляризацією C=0.5)...")
    _model = LogisticRegression(random_state=42, max_iter=1000, C=0.5)
    _model.fit(X_train_combined, y_train)
    print("✅ Model successfully trained with combined features!")

def run_full_model_test(dataset_path="dataset.csv"):
    """
    Тестує модель на відкладених даних та виводить розгорнуту матрицю помилок у термінал.
    """
    global _model, _vectorizer, _scaler
    if _model is None or _vectorizer is None or _scaler is None:
        train_detector(dataset_path)

    df = pd.read_csv(dataset_path, encoding='utf-8')
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    df['cleaned_text'] = df[text_col].apply(clean_text_basic)
    
    _, X_test_raw, _, y_test = train_test_split(
        df['cleaned_text'], df['generated'], test_size=0.2, random_state=42
    )
    _, X_test_orig, _, _ = train_test_split(
        df[text_col], df['generated'], test_size=0.2, random_state=42
    )
    
    X_test_tfidf = _vectorizer.transform(X_test_raw)
    X_test_ling = extract_linguistic_features(X_test_orig)
    X_test_ling_scaled = _scaler.transform(X_test_ling)
    X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_ling_scaled)])
    
    predictions = _model.predict(X_test_combined)
    acc = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=['Human', 'AI'])
    
    print("\n" + "═"*60)
    print("📊 SYSTEM MODEL EVALUATION REPORT (HYBRID MODE)")
    print("═"*60)
    print(f"🎯 Overall Accuracy: {acc * 100:.2f}%")
    print("-" * 60)
    print("📝 Classification Report Detail:")
    print(report)
    print("-" * 60)
    print("🧱 CONFUSION MATRIX:")
    print(f"   -------------------------------------------------------------")
    print(f"   |  Actual Class  |  Predicted HUMAN (0) |   Predicted AI (1) |")
    print(f"   -------------------------------------------------------------")
    print(f"   |  HUMAN (0)     |  {matrix[0][0]:<19} (TH) |  {matrix[0][1]:<18} (FA) |")
    print(f"   |  AI Generated  |  {matrix[1][0]:<19} (FH) |  {matrix[1][1]:<18} (TA) |")
    print(f"   -------------------------------------------------------------")
    print("═"*60 + "\n")

def predict_ai_probability(raw_text):
    global _model, _vectorizer, _scaler
    if _model is None or _vectorizer is None or _scaler is None:
        train_detector()

    cleaned = clean_text_basic(raw_text)
    
    # 1. Слівна частина (TF-IDF)
    tfidf_feat = _vectorizer.transform([cleaned])
    
    # 2. Лінгвістична частина (З нормалізацією масштабу)
    avg_sent_len = calculate_avg_sentence_length(raw_text)
    sent_len_var = calculate_sentence_length_variance(raw_text)
    ttr = calculate_ttr(raw_text)
    adj_density = calculate_adjective_density(raw_text)
    
    ling_feat = np.array([[avg_sent_len, sent_len_var, ttr, adj_density]])
    ling_feat_scaled = _scaler.transform(ling_feat)
    
    # 3. Зшивання докупи
    combined_feat = hstack([tfidf_feat, csr_matrix(ling_feat_scaled)])
    
    # 4. Прогноз ймовірності ШІ
    probabilities = _model.predict_proba(combined_feat)[0]
    ai_prob = probabilities[1] * 100
    return round(ai_prob, 2)