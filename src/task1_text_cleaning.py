"""Task 1: Text Cleaning and Tokenization."""
import re
from typing import List
from nltk.tokenize import word_tokenize


def clean_and_tokenize(text: str) -> List[str]:
    if text is None:
        raise ValueError("Input text cannot be empty")
    if not isinstance(text, str):
        raise TypeError("Input text must be a string")
    if not text.strip():
        raise ValueError("Input text cannot be empty")

    cleaned = re.sub(r"[^\w\s]|_", "", text.lower())

    tokens = word_tokenize(cleaned)
    if not tokens:
        raise ValueError("text contains no words")
    return tokens


if __name__ == "__main__":
    example_text = (
        "Cleans the input text by removing punctuation and converting it to lowercase, "
        "then tokenizes it into individual words."
    )
    print(clean_and_tokenize(example_text))

