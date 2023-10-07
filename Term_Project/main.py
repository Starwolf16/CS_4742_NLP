
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from collections import Counter
from spacy.lang.en import English
from sys import exit
from torch.cuda.amp import GradScaler

nlp = English()

def tokenize(text):
    tokens = [token.text for token in nlp(text)]
    return tokens


dataset = pd.read_csv("emails.csv")

CORPUS = dataset.pop("text")

all_tokens = [token for text in CORPUS.apply(tokenize) for token in text]
vocab = dict(Counter(all_tokens))
vocab_size = len(vocab)

tokens_to_indices = {token: idx for idx, (token, _) in enumerate(vocab.items())}

indexed_data = [CORPUS.apply(lambda text: [tokens_to_indices[token] for token in tokenize(text)])]

# indexed_tensors = [torch.tensor(seq)]

padded_data = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in indexed_data[0]], batch_first=True)
print(type(padded_data))
