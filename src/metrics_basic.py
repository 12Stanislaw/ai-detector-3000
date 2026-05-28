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
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0

    total_words = 0
    for sent in sentences:
        tokens = word_tokenize(sent)
        words = [w for w in tokens if w.isalpha()]
        total_words += len(words)

    avg_length = total_words / len(sentences)
    return round(avg_length, 2)


def calculate_sentence_length_variance(text: str) -> float:
    """
    Рахує дисперсію довжини речень.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return 0.0

    sentence_lengths = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        words = [w for w in tokens if w.isalpha()]
        sentence_lengths.append(len(words))

    variance = np.var(sentence_lengths)
    return round(float(variance), 2)

def calculate_avg_word_length(text: str) -> float:
    """
    Середня довжина слова. ШІ іноді використовує довші або коротші слова в залежності від моделі.
    """
    tokens = word_tokenize(text)
    words = [w for w in tokens if w.isalpha()]
    if not words:
        return 0.0
    
    total_chars = sum(len(w) for w in words)
    return round(total_chars / len(words), 2)

def calculate_punctuation_density(text: str) -> float:
    """
    Щільність розділових знаків.
    """
    if not text:
        return 0.0
    
    punctuation_marks = ",.;:!?"
    count = sum(1 for char in text if char in punctuation_marks)
    return round(count / len(text), 4)

def calculate_sentence_start_diversity(text: str) -> float:
    """
    Різноманітність перших слів у реченнях. 
    ШІ часто починає речення однаково (The, This, It...).
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return 1.0 # Немає статистики
    
    first_words = []
    for sent in sentences:
        words = word_tokenize(sent)
        if words:
            first_words.append(words[0].lower())
    
    if not first_words:
        return 1.0
        
    unique_starts = len(set(first_words))
    return round(unique_starts / len(sentences), 4)

def calculate_comma_density(text: str) -> float:
    """
    Кількість ком на слово. Люди часто використовують складні речення з багатьма комами.
    """
    tokens = word_tokenize(text)
    words = [w for w in tokens if w.isalpha()]
    if not words:
        return 0.0
    
    commas = text.count(',')
    return round(commas / len(words), 4)