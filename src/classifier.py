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
from src.metrics_basic import (
    calculate_avg_sentence_length, 
    calculate_sentence_length_variance,
    calculate_avg_word_length,
    calculate_punctuation_density,
    calculate_sentence_start_diversity,
    calculate_comma_density
)
from src.metrics_advanced import (
    calculate_ttr, 
    calculate_adjective_density,
    calculate_capital_ratio,
    calculate_stopword_density
)

_model = None
_vectorizer = None
_scaler = None

def clean_text_basic(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Очищуємо від сміття, але тримаємо цифри та базову пунктуацію
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_linguistic_features(texts_series):
    """
    Приймає колонку з текстами та повертає матричний масив розширених ознак.
    """
    features_list = []
    for text in texts_series:
        if not isinstance(text, str):
            text = ""
        
        features = [
            calculate_avg_sentence_length(text),
            calculate_sentence_length_variance(text),
            calculate_avg_word_length(text),
            calculate_punctuation_density(text),
            calculate_sentence_start_diversity(text),
            calculate_comma_density(text),
            calculate_ttr(text),
            calculate_adjective_density(text),
            calculate_capital_ratio(text),
            calculate_stopword_density(text)
        ]
        features_list.append(features)
    return np.array(features_list)

def train_detector(dataset_path="dataset.csv"):
    global _model, _vectorizer, _scaler
    
    print("⏳ Завантаження основного датасету...")
    df_orig = pd.read_csv(dataset_path, encoding='utf-8')
    
    # Спочатку розділимо оригінал на train/test, щоб мати чистий тест
    df_train_orig, df_test_orig = train_test_split(df_orig, test_size=0.2, random_state=42, stratify=df_orig['generated'])
    
    # Тепер до тренувальної частини додаємо магію (тільки AI)
    magic_file = "daigt_magic_generations.csv"
    if os.path.exists(magic_file):
        print("🪄 Збагачуємо тренувальний набір різноманітними темами...")
        df_magic = pd.read_csv(magic_file)
        if 'label' in df_magic.columns:
            df_magic = df_magic.rename(columns={'label': 'generated'})
        
        # Додаємо 500 різноманітних AI зразків
        df_ai_extra = df_magic[df_magic['generated'] == 1].sample(n=min(500, len(df_magic)), random_state=42)
        df_train = pd.concat([df_train_orig, df_ai_extra[['text', 'generated']]], ignore_index=True)
        print(f"   Додано {len(df_ai_extra)} AI-зразків до навчання.")
    else:
        df_train = df_train_orig

    text_col = 'text' if 'text' in df_train.columns else df_train.columns[0]
    
    X_train_raw = df_train[text_col].apply(clean_text_basic)
    X_train_orig = df_train[text_col]
    y_train = df_train['generated']

    print("📊 Навчаємо TF-IDF (Generalized mode)...")
    _vectorizer = TfidfVectorizer(
        max_features=2500, 
        min_df=3, 
        max_df=0.7, 
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_train_tfidf = _vectorizer.fit_transform(X_train_raw)
    
    print("📈 Рахуємо лінгвістичні метрики...")
    X_train_ling = extract_linguistic_features(X_train_orig)
    
    print("⚖️ Масштабуємо...")
    _scaler = StandardScaler()
    X_train_ling_scaled = _scaler.fit_transform(X_train_ling)
    
    print("🔗 Зшиваємо ознаки...")
    X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_ling_scaled)])
    
    print("🤖 Навчаємо модель (LogisticRegression C=0.1)...")
    _model = LogisticRegression(random_state=42, max_iter=2000, C=0.1, class_weight='balanced')
    _model.fit(X_train_combined, y_train)
    print("✅ Model successfully trained!")

def run_full_model_test(dataset_path="dataset.csv"):
    """
    Тестує модель на відкладених даних оригінального датасету.
    """
    global _model, _vectorizer, _scaler
    if _model is None or _vectorizer is None or _scaler is None:
        train_detector(dataset_path)

    df_orig = pd.read_csv(dataset_path, encoding='utf-8')
    _, df_test = train_test_split(df_orig, test_size=0.2, random_state=42, stratify=df_orig['generated'])
    
    text_col = 'text' if 'text' in df_test.columns else df_test.columns[0]
    X_test_raw = df_test[text_col].apply(clean_text_basic)
    X_test_orig = df_test[text_col]
    y_test = df_test['generated']
    
    X_test_tfidf = _vectorizer.transform(X_test_raw)
    X_test_ling = extract_linguistic_features(X_test_orig)
    X_test_ling_scaled = _scaler.transform(X_test_ling)
    X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_ling_scaled)])
    
    predictions = _model.predict(X_test_combined)
    acc = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=['Human', 'AI'])
    
    print("\n" + "═"*60)
    print("📊 FINAL MODEL EVALUATION REPORT (VALIDATED ON CLEAN TEST SET)")
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
    tfidf_feat = _vectorizer.transform([cleaned])
    
    features = [
        calculate_avg_sentence_length(raw_text),
        calculate_sentence_length_variance(raw_text),
        calculate_avg_word_length(raw_text),
        calculate_punctuation_density(raw_text),
        calculate_sentence_start_diversity(raw_text),
        calculate_comma_density(raw_text),
        calculate_ttr(raw_text),
        calculate_adjective_density(raw_text),
        calculate_capital_ratio(raw_text),
        calculate_stopword_density(raw_text)
    ]
    
    ling_feat = np.array([features])
    ling_feat_scaled = _scaler.transform(ling_feat)
    
    combined_feat = hstack([tfidf_feat, csr_matrix(ling_feat_scaled)])
    
    probabilities = _model.predict_proba(combined_feat)[0]
    ai_prob = probabilities[1] * 100
    return round(ai_prob, 2)