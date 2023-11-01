import argparse
from nltk import sent_tokenize, word_tokenize
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', type=str)
args = parser.parse_args()

def get_text_from_files(root_dir):
    text_corpus = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                with open(os.path.join(dirpath, filename), 'r', encoding='UTF-8', errors='replace') as file:
                    text = file.read()
                    text_corpus.append(text)
    return text_corpus

documents = get_text_from_files('BBC_News_Summary/News_Articles')
print(f"Number of documents: {len(documents)}")

query = args.query
print(f'Query: {query}')

def preprocess_text(text):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    return [" ".join(sentence) for sentence in words]

vectorizer = TfidfVectorizer()

# Initialize variables to store the most relevant document and its summary
most_relevant_document = None
most_relevant_similarity = -1

# Process each document and find the most relevant one
for document in documents:
    document_vector = vectorizer.fit_transform(preprocess_text(document))
    query_vector = vectorizer.transform(preprocess_text(query))

    similarities = cosine_similarity(query_vector, document_vector).flatten()
    max_similarity = similarities.max()

    if max_similarity > most_relevant_similarity:
        most_relevant_similarity = max_similarity
        most_relevant_document = document

# Generate the summary for the most relevant document
document_vector = vectorizer.fit_transform(preprocess_text(most_relevant_document))
query_vector = vectorizer.transform(preprocess_text(query))
similarities = cosine_similarity(query_vector, document_vector).flatten()
selected_num_sentences = 2
selected_indices = similarities.argsort()[-selected_num_sentences:][::-1]
summary = " ".join([sent_tokenize(most_relevant_document)[i] for i in selected_indices])

# Print the summary
print(f"Summary for the most relevant document (Document {most_relevant_document}):")
print(summary)
