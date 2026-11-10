"""Task 3: Bag of Words (BoW) Implementation."""
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(texts: List[str]) -> pd.DataFrame:
    if not isinstance(texts, list):
        raise TypeError("Input texts must be a list")
    if not texts:
        raise ValueError("Input texts cannot be empty")
    if not all(isinstance(t, str) for t in texts):
        raise ValueError("All texts must be strings")

    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(texts)
    df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return df


if __name__ == "__main__":
    texts = [
        "Converts a list of text strings into a Bag of Words representation",
        "The words that changed the strings",
        "Your duty words of the heart",
        "The Bag full of strings"
    ]
    print(bag_of_words(texts))
