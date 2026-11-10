"""Task 2: Stemming and Lemmatization."""
from typing import Dict, List
from nltk.stem import PorterStemmer, WordNetLemmatizer


def stem_and_lemmatize(tokens: List[str]) -> Dict[str, List[str]]:
    if not isinstance(tokens, list):
        raise TypeError("Input tokens must be a list")
    if not tokens:
        raise ValueError("Input tokens cannot be empty")
    if not all(isinstance(t, str) for t in tokens):
        raise ValueError("All tokens must be strings")

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    stemmed = [stemmer.stem(t) for t in tokens]
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]

    return {
        "original": tokens,
        "stemmed": stemmed,
        "lemmatized": lemmatized
    }


if __name__ == "__main__":
    # Download required NLTK resources (only when running as main)
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    example_tokens = ['goes', 'worlds', 'done', 'being', 'undefined']
    result = stem_and_lemmatize(example_tokens)
    print(f"Original: {result['original']}")
    print(f"Stemmed: {result['stemmed']}")
    print(f"Lemmatized: {result['lemmatized']}")
