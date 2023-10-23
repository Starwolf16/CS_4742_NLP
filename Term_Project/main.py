### IMPORTS ###
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from collections import Counter
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sys import exit
from time import time
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

# Set up Argument Parser
parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, help='Specify the number of epochs for training')
parser.add_argument('-lr', '--learning_rate', type=float, help='Specify the learning rate for training')


### DATA PREPARATION ###
# Initialize the language model for tokenization
nlp = English()

# Define the tokenization function
def tokenize(text):
    tokens = [token.text for token in nlp(text)]
    return tokens

# Read in the dataset 
raw_dataset = pd.read_csv("emails.csv")

# Get the corpus from the dataset
CORPUS = raw_dataset.pop("text")

# Get the Labels from the dataset
labels = raw_dataset.pop("spam")
labels = torch.from_numpy(labels.to_numpy())

# Tokenize the text
all_tokens = [token for text in CORPUS.apply(tokenize) for token in text]

# Get the vocab of the corpus
vocab = dict(Counter(all_tokens))
vocab_size = len(vocab)

# Convert the tokens to indices
tokens_to_indices = {token: idx for idx, (token, _) in enumerate(vocab.items())}
indexed_data = [CORPUS.apply(lambda text: [tokens_to_indices[token] for token in tokenize(text)])]

# Pad the data since all the samples are not the same length
padded_data = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in indexed_data[0]], batch_first=True)

# Split the data for training and testing
DATA_train, DATA_test, labels_train, labels_test = train_test_split(
    padded_data, 
    labels,
    train_size=.7, 
    random_state=42, 
    stratify=labels
)

# Define Dataset for data encapsulation
class text_dataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, labels_tensor: torch.Tensor) -> None:
        self.data = data_tensor
        self.labels = labels_tensor
    
    def __getitem__(self, index):
        return [self.data[index], self.labels[index]]
    
    def __len__(self):
        return(len(labels))
    
# Instantiate Dataset instances
train_set = text_dataset(DATA_train, labels_train)
test_set = text_dataset(DATA_test, labels_test)

# Instantiate the dataloaders
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=5, shuffle=True)


### TRAINING PHASE ###

start_time = time()

