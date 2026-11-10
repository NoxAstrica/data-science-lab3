"""Task 7: Integrated NLP Pipeline."""
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.task1_text_cleaning import clean_and_tokenize
from src.task2_stemm_lemma import stem_and_lemmatize
from src.task5_word2vec import train_word2vec
from src.task6_Ngrams import generate_ngrams, count_ngrams


def analyze_documents(
    documents: List[str],
    use_lemmatization: bool = True,
    ngram_range: Tuple[int, int] = (1, 2),
    vector_method: str = "tfidf"
) -> Dict[str, Any]:
    # Input validation
    if not isinstance(documents, list):
        raise TypeError("Documents must be a list.")
    if not documents:
        raise ValueError("Documents cannot be empty.")
    if not all(isinstance(doc, str) for doc in documents):
        raise TypeError("All documents must be strings.")
    if len(documents) < 2:
        raise ValueError("At least 2 documents are required.")
    if vector_method not in ["bow", "tfidf"]:
        raise ValueError("vector_method must be either 'bow' or 'tfidf'.")
    if not isinstance(ngram_range, tuple) or len(ngram_range) != 2 or not all(isinstance(i, int) for i in ngram_range) or ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
        raise ValueError("Invalid ngram_range.")

    # Step 1: Preprocess documents
    preprocessed_docs = []
    all_tokens = []
    
    for doc in documents:
        cleaned_tokens = clean_and_tokenize(doc)
        # stem_and_lemmatize returns a dictionary, e.g., {"lemmatized": [...], "stemmed": [...]}
        token_results = stem_and_lemmatize(cleaned_tokens)
        # Select the appropriate list based on the parameter
        processed_tokens = token_results["lemmatized"] if use_lemmatization else token_results["stemmed"]
        preprocessed_docs.append(processed_tokens)
        all_tokens.extend(processed_tokens)
    
    # Calculate preprocessing statistics
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    avg_doc_length = total_tokens / len(documents) if documents else 0
    
    # Step 2: Vectorization
    reconstructed_texts = [" ".join(tokens) for tokens in preprocessed_docs]
    
    if vector_method == "bow":
        vectorizer = CountVectorizer(ngram_range=ngram_range)
    else:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        
    feature_matrix_array = vectorizer.fit_transform(reconstructed_texts).toarray()
    feature_names = vectorizer.get_feature_names_out()
    
    feature_df = pd.DataFrame(
        feature_matrix_array,
        columns=feature_names,
        index=[f"doc_{i}" for i in range(len(documents))]
    )
    
    # Step 3: Word embeddings
    # Filter out empty documents for Word2Vec
    w2v_sentences = [tokens for tokens in preprocessed_docs if tokens]
    if not w2v_sentences:
        w2v_sentences = [[""] ]
    
    word2vec_model = train_word2vec(w2v_sentences)
    vocabulary_size = len(word2vec_model.wv)
    vector_size = word2vec_model.vector_size
    
    # Step 4: N-gram analysis
    # Combine all documents for n-gram analysis
    combined_tokens = [token for doc in preprocessed_docs for token in doc]
    
    # Generate bigrams and trigrams
    bigrams_list = list(generate_ngrams(" ".join(combined_tokens), 2)) if len(combined_tokens) >= 2 else []
    trigrams_list = list(generate_ngrams(" ".join(combined_tokens), 3)) if len(combined_tokens) >= 3 else []
    
    bigram_counts = count_ngrams(bigrams_list) if bigrams_list else {}
    trigram_counts = count_ngrams(trigrams_list) if trigrams_list else {}
    
    top_bigrams = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)
    top_trigrams = sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True)
    
    bigram_diversity = len(bigram_counts) / len(bigrams_list) if bigrams_list else 0.0
    
    # Step 5: Document similarity
    sim_array = cosine_similarity(feature_matrix_array)
    similarity_df = pd.DataFrame(
        sim_array,
        index=[f"doc_{i}" for i in range(len(documents))],
        columns=[f"doc_{i}" for i in range(len(documents))]
    )

    # Compile results
    results = {
        "preprocessed": {
            "documents": preprocessed_docs,
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "avg_doc_length": round(avg_doc_length, 2)
        },
        "vectorization": {
            "method": vector_method,
            "feature_matrix": feature_df,
            "feature_names": list(feature_names),
            "shape": feature_df.shape
        },
        "word_embeddings": {
            "model": word2vec_model,
            "vocabulary_size": vocabulary_size,
            "vector_size": vector_size
        },
        "ngram_analysis": {
            "top_bigrams": top_bigrams,
            "top_trigrams": top_trigrams,
            "bigram_diversity": round(bigram_diversity, 3)
        },
        "document_similarity": similarity_df
    }
    
    return results


def find_similar_documents(
    doc_index: int,
    similarity_matrix: pd.DataFrame,
    top_n: int = 3
) -> List[Tuple[int, float]]:
    if not isinstance(similarity_matrix, pd.DataFrame):
        raise TypeError("similarity_matrix must be a pandas DataFrame")
    if doc_index < 0 or doc_index >= len(similarity_matrix):
        raise ValueError("doc_index must be between")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    
    doc_name = similarity_matrix.index[doc_index]
    
    # Get similarity scores for the document (excluding itself), sorted top first
    similarities = similarity_matrix.loc[doc_name].drop(doc_name)
    top_similarities = similarities.sort_values(ascending=False).head(top_n)
    
    result = []
    for name, score in top_similarities.items():
        doc_idx = int(name.split('_')[1])
        result.append((doc_idx, round(float(score), 4)))
    
    return result


def get_document_keywords(
    doc_index: int,
    feature_matrix: pd.DataFrame,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    if not isinstance(feature_matrix, pd.DataFrame):
        raise TypeError("feature_matrix must be a pandas DataFrame")
    if doc_index < 0 or doc_index >= len(feature_matrix):
        raise ValueError("doc_index must be between")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    
    # Get feature scores for the document
    doc_scores = feature_matrix.iloc[doc_index]
    
    # Sort by score (descending) and get top N
    top_features = doc_scores.sort_values(ascending=False).head(top_n)
    
    # Convert to list of tuples
    result = [(feature, round(score, 4)) for feature, score in top_features.items()]
    
    return result


def compare_vectorization_methods(
    documents: List[str]
) -> Dict[str, pd.DataFrame]:
    if not isinstance(documents, list):
        raise TypeError("Documents must be a list.")
    if not documents:
        raise ValueError("Documents cannot be empty.")
    if not all(isinstance(doc, str) for doc in documents):
        raise TypeError("All documents must be strings.")
    if len(documents) < 2:
        raise ValueError("At least 2 documents are required")
    
    # Run analysis with both methods
    bow_results = analyze_documents(documents, vector_method="bow")
    tfidf_results = analyze_documents(documents, vector_method="tfidf")
    
    return {
        "bow": bow_results['vectorization']['feature_matrix'],
        "tfidf": tfidf_results['vectorization']['feature_matrix']
    }


if __name__ == "__main__":
    # Example usage
    sample_documents = [
        "Natural language processing is amazing and fun to learn.",
        "Machine learning and NLP are related fields in AI.",
        "Deep learning models are powerful for NLP tasks.",
        "Python is great for machine learning and data science."
    ]
    
    print("=" * 80)
    print("INTEGRATED NLP PIPELINE ANALYSIS")
    print("=" * 80)
    
    # Run comprehensive analysis
    results = analyze_documents(
        sample_documents,
        use_lemmatization=True,
        ngram_range=(1, 2),
        vector_method="tfidf"
    )
    
    # Display results
    print("\n1. PREPROCESSING STATISTICS")
    print("-" * 80)
    print(f"Total tokens: {results['preprocessed']['total_tokens']}")
    print(f"Unique tokens: {results['preprocessed']['unique_tokens']}")
    print(f"Average document length: {results['preprocessed']['avg_doc_length']}")
    
    print("\n2. VECTORIZATION")
    print("-" * 80)
    print(f"Method: {results['vectorization']['method'].upper()}")
    print(f"Feature matrix shape: {results['vectorization']['shape']}")
    print(f"Number of features: {len(results['vectorization']['feature_names'])}")
    
    print("\n3. WORD EMBEDDINGS")
    print("-" * 80)
    print(f"Vocabulary size: {results['word_embeddings']['vocabulary_size']}")
    print(f"Vector dimensionality: {results['word_embeddings']['vector_size']}")
    
    print("\n4. N-GRAM ANALYSIS")
    print("-" * 80)
    print(f"Top 5 bigrams:")
    for bigram, count in results['ngram_analysis']['top_bigrams'][:5]:
        print(f"  {bigram}: {count}")
    print(f"Bigram diversity: {results['ngram_analysis']['bigram_diversity']}")
    
    print("\n5. DOCUMENT SIMILARITY")
    print("-" * 80)
    print("Similarity matrix:")
    print(results['document_similarity'])
    
    print("\n6. SIMILAR DOCUMENTS TO DOC_0")
    print("-" * 80)
    similar_docs = find_similar_documents(0, results['document_similarity'], top_n=2)
    for doc_idx, similarity in similar_docs:
        print(f"  Document {doc_idx}: {similarity:.4f}")
    
    print("\n7. TOP KEYWORDS IN DOC_0")
    print("-" * 80)
    keywords = get_document_keywords(0, results['vectorization']['feature_matrix'], top_n=5)
    for keyword, score in keywords:
        print(f"  {keyword}: {score:.4f}")
    
    print("\n8. COMPARISON: BOW vs TF-IDF")
    print("-" * 80)
    comparison = compare_vectorization_methods(sample_documents[:3])
    print("BoW shape:", comparison['bow'].shape)
    print("TF-IDF shape:", comparison['tfidf'].shape)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
