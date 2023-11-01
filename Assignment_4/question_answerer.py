import argparse
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-q', '--question', type=str)

args = parser.parse_args()

# Define the directory containing BBC News articles and subfolders
document_directory = "BBC_News_Summary/News_Articles/"
os.listdir(document_directory)

def process_question(question):
    # Tokenize the question
    tokens = word_tokenize(question)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Perform stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    return stemmed_tokens

def retrieve_documents(question_tokens):
    print('Retrieving Documents')
    documents = []

    # Recursively traverse the directory and its subdirectories
    for root, dirs, files in os.walk(document_directory):

        for filename in tqdm(files):
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                
                # Read the content of the document
                with open(file_path, 'r', encoding='iso-8859-1') as file:
                    document_text = file.read()
                
                # Tokenize the document text
                document_tokens = word_tokenize(document_text)
                
                # Remove stopwords and apply stemming
                stop_words = set(stopwords.words('english'))
                filtered_tokens = [token for token in document_tokens if token.lower() not in stop_words]
                stemmer = PorterStemmer()
                stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

                # Compute a similarity score between the question and document
                similarity = calculate_similarity(question_tokens, stemmed_tokens)

                
                # Set Similarity Threshold
                if similarity > 0.2:  
                    documents.append(document_text)

    return documents


def calculate_similarity(question_tokens, document_tokens):
    # Vectorize the question and document
    tfidf_vectorizer = TfidfVectorizer()
    question_vector = tfidf_vectorizer.fit_transform([" ".join(question_tokens)])
    document_vector = tfidf_vectorizer.transform([" ".join(document_tokens)])
    
    # Compute cosine similarity between the question and document
    similarity = cosine_similarity(question_vector, document_vector)[0][0]
    
    return similarity

def select_answer(question, documents):
    if not documents:
        return "No relevant documents found."

    # Tokenize the question and preprocess it
    question_tokens = process_question(question)

    # Initialize variables to store the best answer and its matching score
    best_answer = None
    best_score = 0  

    for document in documents:
        # Tokenize and preprocess the document
        document_tokens = word_tokenize(document)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in document_tokens if token.lower() not in stop_words]
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

        # Calculate the matching score based on keyword matching
        keyword_match_score = calculate_keyword_match(question_tokens, stemmed_tokens)

        # Update the best answer if the current document has a higher score
        if keyword_match_score > best_score:
            best_score = keyword_match_score
            best_answer = document

    if best_answer:
        return best_answer
    else:
        return "No answer found in the documents."

def calculate_keyword_match(question_tokens, document_tokens):
    # Calculate the keyword matching score based on common tokens
    common_tokens = set(question_tokens).intersection(document_tokens)
    match_score = len(common_tokens) / len(question_tokens)
    return match_score


def extract_answer_from_document(document):
    # Split the document into sentences
    sentences = nltk.sent_tokenize(document)

    if sentences:
        # Extract the first sentence as the answer
        return sentences[0]
    else:
        return "No answer found in the document."


question = args.question
question_tokens = process_question(question)
documents = retrieve_documents(question_tokens)
print(len(documents))
answer = select_answer(question, documents)
print("Question:", question)
print("Answer:", answer)
