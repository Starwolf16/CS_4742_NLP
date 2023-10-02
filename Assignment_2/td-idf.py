import string
import numpy as np

with open('Assignment_2\sample_txt.txt') as content:
    text = content.read()

    text = text.strip('\n')
    text = text.translate(str.maketrans('','',string.punctuation))

    text = np.array(text.split())

tokens, counts = np.unique(text, return_counts=True)
word_counts = dict(zip(tokens, counts))

total_tokens = len(tokens)

print(f'\nTokens: {tokens}\n')
print(f'Vocab: {tokens}\n')
for key, value in word_counts.items():
    print(f'TF for \'{key}\': {value} / {total_tokens} = {(value / total_tokens):.2f}')

print('\nIDF for all tokens is 1 since all lines are considered as 1 document')
print('Therefor the TF-IDF Vector for each token is the same as the TF Value\n')



