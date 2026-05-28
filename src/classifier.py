import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Імпортуємо розширені метрики
from src.metrics_basic import (
    calculate_avg_sentence_length, 
    calculate_sentence_length_variance,
    calculate_avg_word_length,
    calculate_punctuation_density,
    calculate_sentence_start_diversity,
    calculate_comma_density,
    calculate_dialogue_density,
    calculate_sentence_end_diversity
)
from src.metrics_advanced import (
    calculate_adjective_density,
    calculate_capital_ratio
)

_model = None
_vectorizer = None
_scaler = None

def extract_hybrid_features(text):
    words = text.split()
    word_count = len(words)
    if word_count == 0: return [0] * 9
    
    avg_len = calculate_avg_sentence_length(text)
    variance = calculate_sentence_length_variance(text)
    
    return [
        avg_len,
        variance,
        calculate_avg_word_length(text),
        calculate_punctuation_density(text),
        calculate_sentence_start_diversity(text),
        calculate_comma_density(text),
        calculate_dialogue_density(text),
        calculate_sentence_end_diversity(text),
        calculate_capital_ratio(text)
    ]

def train_detector(dataset_path="dataset.csv"):
    global _model, _vectorizer, _scaler

    print("⏳ Навчання фінальної гібридної моделі...")
    df = pd.read_csv(dataset_path)
    
    # Використовуємо символьні тріграми (текстура)
    _vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 3),
        max_features=300,
        min_df=5
    )
    X_char = _vectorizer.fit_transform(df['text'].apply(lambda x: x.lower()))
    
    X_ling = np.array([extract_hybrid_features(t) for t in df['text']])
    _scaler = StandardScaler()
    X_ling_scaled = _scaler.fit_transform(X_ling)
    
    X_combined = hstack([X_char, csr_matrix(X_ling_scaled)])
    
    # Логістична регресія з дуже сильною регуляризацією (C=0.1)
    # Це робить модель менш впевненою, що нам і потрібно
    _model = LogisticRegression(C=0.1, class_weight='balanced', random_state=42)
    _model.fit(X_combined, df['generated'])
    print("✅ Модель готова!")

def predict_ai_probability(raw_text):
    global _model, _vectorizer, _scaler
    if _model is None:
        train_detector()

    words = raw_text.split()
    if len(words) < 15:
        return 10.0

    # 1. Екстракція
    char_feat = _vectorizer.transform([raw_text.lower()])
    ling_feat = np.array([extract_hybrid_features(raw_text)])
    ling_scaled = _scaler.transform(ling_feat)
    combined = hstack([char_feat, csr_matrix(ling_scaled)])
    
    # 2. Прогноз (базовий)
    prob = _model.predict_proba(combined)[0][1] * 100
    
    # 3. ЕВРИСТИКИ "ЖИВОЇ ЛЮДИНИ"
    
    # А. Емоції та лапки (Діалоги - це 100% людина)
    dialogue = calculate_dialogue_density(raw_text)
    emotions = calculate_sentence_end_diversity(raw_text)
    if dialogue > 0.01: prob -= 30
    if emotions > 0.2: prob -= 20
        
    # Б. Складність слів (ШІ пише в діапазоні 4.5 - 6.0)
    avg_word = calculate_avg_word_length(raw_text)
    if avg_word < 4.2 or avg_word > 6.5:
        prob -= 20
        
    # В. Нерівномірність речень (ШІ пише рівно)
    var = calculate_sentence_length_variance(raw_text)
    if var > 150: prob -= 25
    if var < 15: prob += 15 # Дуже рівний текст підозрілий

    # Г. "Проста мова" (короткі речення)
    avg_sent = calculate_avg_sentence_length(raw_text)
    if avg_sent < 12: prob -= 20 # Занадто проста мова для ШІ-есе
    
    # Обмеження
    prob = min(98.0, max(5.0, prob))
    
    # Калібрування "сірої зони"
    if 40 < prob < 70:
        prob *= 0.7 # Схиляємось до безпеки
        
    return round(prob, 2)

def run_full_model_test(dataset_path="dataset.csv"):
    train_detector(dataset_path)
