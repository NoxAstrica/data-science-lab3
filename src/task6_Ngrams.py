"""Task 6: N-gram Generation and Frequency Counting."""
from typing import List, Tuple, Dict
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize


def generate_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("N must be an integer")
    if not text.strip():
        raise ValueError("Text cannot be empty")
    if n < 1:
        raise ValueError("N must be at least 1")

    tokens = word_tokenize(text)
    n_grams = list(ngrams(tokens, n))
    return n_grams


def count_ngrams(ngrams_list: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], int]:
    if not isinstance(ngrams_list, list):
        raise TypeError("N-grams list must be a list")
    if not ngrams_list:
        raise ValueError("N-grams list cannot be empty")

    ngram_counts = Counter(ngrams_list)
    return dict(ngram_counts)


if __name__ == "__main__":
    # Download required NLTK resources (only when running as main)
    import nltk
    nltk.download('punkt', quiet=True)

    text = "I love NLP and learning NLP techniques"

    # Generate bigrams
    bigrams = generate_ngrams(text, 2)
    print("Bigrams:", bigrams)

    # Count bigrams
    bigram_counts = count_ngrams(bigrams)
    print("\nBigram counts:", bigram_counts)
    
    # Generate trigrams
    trigrams = generate_ngrams(text, 3)
    print("\nTrigrams:", trigrams)
