import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# --- АВТОМАТИЧНЕ ЗАВАНТАЖЕННЯ РЕСУРСІВ NLTK ---
required_resources = {
    'tokenizers/punkt': 'punkt',
    'tokenizers/punkt_tab': 'punkt_tab'
}

for path, package in required_resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package, quiet=True)
# ---------------------------------------------

def calculate_avg_sentence_length(text: str) -> float:
    """
    Рахує середню довжину речення в тексті (кількість слів на одне речення).
    ШІ зазвичай тримає цей показник у межах 'золотої середини' (15-20 слів).
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    # 1. Розбиваємо текст на речення
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0

    total_words = 0
    # 2. Рахуємо слова в кожному реченні
    for sent in sentences:
        tokens = word_tokenize(sent)
        words = [w for w in tokens if w.isalpha()]
        total_words += len(words)

    # 3. Знаходимо середнє
    avg_length = total_words / len(sentences)
    return round(avg_length, 2)


def calculate_sentence_length_variance(text: str) -> float:
    """
    Рахує дисперсію (variance) довжини речень.
    Показує, наскільки сильно довжина речень коливається.
    У людини цей показник ВІДЧУТНО вищий через 'рваний' стиль мовлення.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    sentences = sent_tokenize(text)
    # Якщо речення одне або їх немає, коливання (дисперсії) не існує
    if len(sentences) <= 1:
        return 0.0

    sentence_lengths = []
    
    # 1. Збираємо довжину кожного речення в список
    for sent in sentences:
        tokens = word_tokenize(sent)
        words = [w for w in tokens if w.isalpha()]
        sentence_lengths.append(len(words))

    # 2. Використовуємо numpy для швидкого розрахунку дисперсії (статистичного розкиду)
    variance = np.var(sentence_lengths)
    
    return round(float(variance), 2)