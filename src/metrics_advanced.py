import nltk
from nltk.tokenize import word_tokenize

# --- АВТОМАТИЧНЕ ЗАВАНТАЖЕННЯ РЕСУРСІВ ДЛЯ НОВИХ ВЕРСІЙ NLTK ---
required_nltk_resources = {
    'tokenizers/punkt': 'punkt',
    'tokenizers/punkt_tab': 'punkt_tab',  # <-- Ось цей хлопець вилітав!
    'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
    'taggers/averaged_perceptron_tagger_eng': 'averaged_perceptron_tagger_eng' # Теж потрібно для нових версій
}

for path, package in required_nltk_resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        print(f"📥 NLTK resource not found. Downloading package: {package}...")
        nltk.download(package)

#---TTR_score =  (number of unique words) / (total number of words)---
def calculate_ttr(text):

    text= text.lower()
    
    # 1. Splitting text into individual words (tokens) using NLTK
    tokens = word_tokenize(text)
    
    #2. Remove punctuation (keep only letters and numbers)
    words = [token for token in tokens if token.isalnum()]
    
    # Prevent division by zero if the text is empty
    if len(words) == 0:
        return 0.0
    
    #3. Count number of unique total words and find TTR_score   
    total_words = len(words)
    unique_words = len(set(words))
    ttr_score = unique_words / total_words
    
    #4. Return score(round to 4 digits) 
    return round(ttr_score, 4)


#---Показує, який відсоток від усіх слів займають описові слова (JJ, JJR, JJS)---
def calculate_adjective_density(text):

    tokens = word_tokenize(text.lower())
    
    words = [token for token in tokens if token.isalnum()]
    
    if len(words) == 0:
        return 0.0
        
    # NLTK parses a list of words and returns tuples of the form: [(‘love’, ‘NN’), (‘is’, ‘VBZ’), (‘great’, ‘JJ’)]
    tagged_words = nltk.pos_tag(words)
    
    # Counting the number of occurrences of tags starting with ‘JJ’ (adjectives)
    adj_count = sum(1 for word, tag in tagged_words if tag.startswith('JJ'))
    
    # Calculate the percentage of adjectives out of the total number of words
    density = adj_count / len(words)
    
    return round(density, 4)
