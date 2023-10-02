import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample text corpus
corpus = [
    "It is going to rain today.",
    "Today, I am not going outside.",
    "NLP is an interesting topic.",
    "NLP includes ML, DL topics too.",
    "I am going to complete NLP homework, today.",
    "NLP homework is comprehensive.",
    "I am going outside to meet my friends."
]

# Step 1: Calculate TF-IDF Vectors

# Create a TF-IDF vectorizer object
tfidf_vectorizer = TfidfVectorizer()

# Calculate TF-IDF vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Get the vocabulary (unique words) and their indices
vocab = tfidf_vectorizer.get_feature_names_out()

# Step 2: Calculate Co-occurrence Matrix

# Create a vocabulary of unique words
tokens = ' '.join(corpus).split()
vocab_co_occurrence = list(set(tokens))

# Create a co-occurrence matrix
co_occurrence_matrix = np.zeros((len(vocab_co_occurrence), len(vocab_co_occurrence)))

# Define the context window size (you can adjust this as needed)
window_size = 2

# Iterate through the text to fill the co-occurrence matrix
for i in range(len(tokens)):
    target_word = tokens[i]
    target_index = vocab_co_occurrence.index(target_word)
    
    # Define the context window
    start = max(0, i - window_size)
    end = min(len(tokens), i + window_size + 1)
    
    # Count co-occurrences within the context window
    for j in range(start, end):
        if j != i:
            context_word = tokens[j]
            context_index = vocab_co_occurrence.index(context_word)
            co_occurrence_matrix[target_index][context_index] += 1

# Step 3: Calculate Similarity for Each Pair and Print Top 5

# Create a list to store similarity scores and word pairs
similarity_scores = []

# Iterate through all unique word pairs
for i in range(len(vocab_co_occurrence)):
    for j in range(i + 1, len(vocab_co_occurrence)):
        word_pair = (vocab_co_occurrence[i], vocab_co_occurrence[j])

        # Find the indices of the selected words in the TF-IDF vocabulary
        index_word_1 = np.where(vocab == word_pair[0])
        index_word_2 = np.where(vocab == word_pair[1])

        # Calculate the cosine similarity for the selected word pair using TF-IDF vectors
        similarity_tfidf = cosine_similarity(tfidf_matrix[:, [index_word_1, index_word_2]])
        similarity_scores.append((word_pair, similarity_tfidf[0][0]))  # Use [0][0] to get the single similarity value

# Sort the similarity scores in descending order
sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# Print the top 5 similar word pairs
for i, (word_pair, similarity_score) in enumerate(sorted_similarity_scores[:5]):
    print(f"{i+1}. Similarity between '{word_pair[0]}' and '{word_pair[1]}': {similarity_score:.4f}")
