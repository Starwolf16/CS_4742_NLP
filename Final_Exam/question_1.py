import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define two documents
doc1 = nlp("This is an example sentence.")
doc2 = nlp("This is another example sentence.")

# Calculate the similarity between the two documents
similarity = doc1.similarity(doc2)

print(f"Document similarity: {similarity}")
print(nlp.pipe_names)