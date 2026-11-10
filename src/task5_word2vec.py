"""Task 5: Word2Vec Model Training."""
from typing import List, Tuple
from gensim.models import Word2Vec


def train_word2vec(
    sentences: List[List[str]], 
    vector_size: int = 100, 
    window: int = 5, 
    min_count: int = 1, 
    workers: int = 4
) -> Word2Vec:
    if not isinstance(sentences, list):
        raise TypeError("Sentences must be a list.")
    if not sentences:
        raise ValueError("Sentences cannot be empty.")
    for sentence in sentences:
        if not isinstance(sentence, list):
            raise TypeError("Sentences must be a list of lists.")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model


def find_similar_words(model: Word2Vec, word: str, topn: int = 10) -> List[Tuple[str, float]]:
    if not isinstance(model, Word2Vec):
        raise TypeError("Model must be a Word2Vec instance.")
    if not isinstance(word, str):
        raise TypeError("Word must be a string.")
    if word not in model.wv:
        raise KeyError(f"Word '{word}' is not in vocabulary.")
    return model.wv.most_similar(word, topn=topn)


if __name__ == "__main__":
    # Download required NLTK resources (only when running as main)
    import nltk
    nltk.download('punkt', quiet=True)

    # Define example sentences
    example_sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun'],
        ['nlp', 'can', 'be', 'challenging'],
        ['artificial', 'intelligence', 'is', 'related', 'to', 'nlp']
    ]

    # Train Word2Vec model on the sentences
    model = train_word2vec(example_sentences)

    # Find and display similar words
    similar_words = find_similar_words(model, 'nlp')
    print("Words similar to 'nlp':")
    for word, score in similar_words:
        print(f"  {word}: {score:.4f}")

    similar_words = find_similar_words(model, 'learning')
    print("\nWords similar to 'learning':")
    for word, score in similar_words:
        print(f"  {word}: {score:.4f}")
