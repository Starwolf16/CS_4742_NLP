import numpy as np

# Define the text as one document
text = "It is going to rain today. Today, I am not going outside. NLP is an interesting topic. NLP includes ML, DL topics too. I am going to complete NLP homework, today. NLP homework is comprehensive. I am going outside to meet my friends."

# Tokenize the text into words
tokens = text.split()

# Create a vocabulary of unique words
vocab = list(set(tokens))

# Create a co-occurrence matrix
co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))

# Define the context window size
window_size = 1

# Iterate through the text to fill the co-occurrence matrix
for i in range(len(tokens)):
    target_word = tokens[i]
    target_index = vocab.index(target_word)
    
    # Define the context window
    start = max(0, i - window_size)
    end = min(len(tokens), i + window_size + 1)
    
    # Count co-occurrences within the context window
    for j in range(start, end):
        if j != i:
            context_word = tokens[j]
            context_index = vocab.index(context_word)
            co_occurrence_matrix[target_index][context_index] += 1

# Save the co-occurrence matrix to a TSV file with words on top and left
output_file = "co_occurrence_matrix.tsv"
with open(output_file, "w") as f:
    # Write the header row with words on top
    f.write("\t" + "\t".join(vocab) + "\n")
    
    # Write the matrix with words on the left
    for i in range(len(vocab)):
        f.write(vocab[i] + "\t" + "\t".join(map(str, co_occurrence_matrix[i])) + "\n")
