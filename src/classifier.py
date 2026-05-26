import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

_model = None
_vectorizer = None

def clean_text_basic(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яіієєґґ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_detector(dataset_path="dataset.csv"):
    global _model, _vectorizer
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset path {dataset_path} not found!")

    print("⏳ Loading and preparing dataset for training...")
    df = pd.read_csv(dataset_path, encoding='utf-8')
    
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    df['cleaned_text'] = df[text_col].apply(clean_text_basic)
    
    X = df['cleaned_text']
    y = df['generated']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("📊 Vectorizing texts (TF-IDF with Overfitting prevention)...")
    _vectorizer = TfidfVectorizer(
        max_features=1000,      
        min_df=3,               
        ngram_range=(1, 2)      
    )
    X_train_tfidf = _vectorizer.fit_transform(X_train)
    
    print("🤖 Training Logistic Regression model...")
    _model = LogisticRegression(random_state=42)
    _model.fit(X_train_tfidf, y_train)
    print("✅ Model successfully trained and stored in memory.")

def run_full_model_test(dataset_path="dataset.csv"):
    """
    Tests the model on the validation set and prints a comprehensive
    Confusion Matrix and Classification Report in English.
    """
    global _model, _vectorizer
    
    if _model is None or _vectorizer is None:
        print("⚠️ Model not trained. Please run train_detector() first.")
        return

    df = pd.read_csv(dataset_path, encoding='utf-8')
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    df['cleaned_text'] = df[text_col].apply(clean_text_basic)
    
    _, X_test, _, y_test = train_test_split(
        df['cleaned_text'], df['generated'], test_size=0.2, random_state=42
    )
    
    X_test_tfidf = _vectorizer.transform(X_test)
    predictions = _model.predict(X_test_tfidf)
    
    # Calculate metrics
    acc = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=['Human', 'AI'])
    
    # Print beautiful logs in English for screenshots
    print("\n" + "═"*60)
    print("📊 SYSTEM MODEL EVALUATION REPORT (MVP Stage)")
    print("═"*60)
    print(f"🎯 Overall Accuracy: {acc * 100:.2f}%")
    print(f"   [Successfully classified {matrix[0][0] + matrix[1][1]} out of {len(y_test)} test samples]")
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
    print("\n💡 Matrix Legend:")
    print(f"   • TH (True Human): {matrix[0][0]} texts correctly identified as human-written.")
    print(f"   • TA (True AI):    {matrix[1][1]} texts correctly identified as AI-generated.")
    print(f"   • FA (False AI):   {matrix[0][1]} human texts mistakenly flagged as AI (Type I Error).")
    print(f"   • FH (False Human): {matrix[1][0]} AI texts that bypassed the detector (Type II Error).")
    print("═"*60 + "\n")

def predict_ai_probability(raw_text):
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        train_detector()

    cleaned = clean_text_basic(raw_text)
    vectorized = _vectorizer.transform([cleaned])
    probabilities = _model.predict_proba(vectorized)[0]
    ai_prob = probabilities[1] * 100
    return round(ai_prob, 2)
