### IMPORTS ###
import argparse
import os

# Define file type checkers
def model_type_check(file):
    allowed_type = '.pth'

    filename, file_extension = os.path.splitext(file)
    if file_extension != allowed_type:
        parser.error('Please ensure your checkpoint file is of type \'.pth\'')

    return file


# Set up Argument Parser
parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='Specify the number of epochs for training. Default == 10')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='Specify the learning rate for training. Default == 0.001')
parser.add_argument('-c', '--checkpoint', type=lambda checkpoint_file:model_type_check(checkpoint_file),
                    help='Specify a file to use as a model checkpoint. Must be of type .pth')
parser.add_argument('-g', '--graph', action='store_true', default=True,
                    help='Print out a graph of the losses over epochs')

args = parser.parse_args()

import autoencoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.amp
import torch.optim as optim

from collections import Counter
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sys import exit
from time import time
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'-----\nRunning PyTorch v{torch.__version__} on {device} device\n-----\n')

num_epochs = args.epochs
learning_rate = args.learning_rate


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

# Split off the spam emails 
ham_emails = padded_data[labels == 0]
spam_emails = padded_data[labels == 1]

ham_labels = labels[labels == 0]
spam_labels = labels[labels == 1]


# Split the data for training and testing
DATA_train, DATA_test, _, labels_test = train_test_split(
    ham_emails,
    ham_labels,
    train_size=.85, 
    random_state=42, 
)


# Define Datasets for data encapsulation
class text_dataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor) -> None:
        self.data = data_tensor
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self) -> int:
        return(self.data.size()[0])
    
class test_dataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, labels_tensor: torch.Tensor) -> None:
        self.data = data_tensor
        self.labels = labels_tensor
    
    def __getitem__(self, index):
        return [self.data[index], self.labels[index]]
    
    def __len__(self) -> int:
        return(self.data.size()[0])
    

# Combine the testing Ham data with the spam data
DATA_test = torch.concat([DATA_test, spam_emails])
labels_test = torch.concat([labels_test, spam_labels])

# Instantiate Dataset instances
train_set = text_dataset(DATA_train)
test_set = test_dataset(DATA_test, labels_test)


# Instantiate the dataloaders
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)


### MODEL SET-UP ###

if not args.checkpoint:
    model = autoencoder.AutoEncoder(padded_data.shape[1], 0)
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()
else: 
    kwargs, model_params = torch.load(args.checkpoint)
    model = autoencoder.AutoEncoder(kwargs)
    model.load_state_dict(model_params)

# Move the model to the GPU if available
if device == 'cuda':
    model = model.to(device)


### TRAINING PHASE ###
if not args.checkpoint:
    print(model)
    print(optimizer)

    # Save model and optimizer configs
    with open('model_and_optim_config.txt', 'w') as config_file:
        print(model, optimizer, file=config_file)

    loss_list = list()

    start_time = time()

    for epoch in tqdm(range(num_epochs)):
        for emails in train_loader:
            
            # Move the emails tensor to cuda 
            emails = emails.to(device, torch.float)
            
            # Use the torch.autocast context manager to use AMP for tensor cores
            with torch.autocast(device_type=device):

                # Run the emails through the Autoencoder newtork
                output = model(emails)

                # Calculate the Loss
                loss = loss_fn(output, emails)

            # Use GradScaler to backpropoate the loss 
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_list.append(loss.item())


    print('\n\n----- TRAINING STATS -----')
    print(f'Training Time: {(time() - start_time):.2f} secs')
    print(f'Maximum Loss: {np.max(loss_list):.3f}')
    print(f'Minimum Loss: {np.min(loss_list):.3f}')
    print(f'Average Loss: {np.average(loss_list):.3f}')


    if args.graph:
        plt.figure()
        plt.plot([y for y in range(num_epochs)], loss_list)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('loss_plot.svg')


### TESTING PHASE ###

recon_error_list = []
labels_list = []

model.eval()
with torch.no_grad():
    for emails, label in test_loader:
        emails = emails.to(device, torch.float)

        with torch.autocast(device_type=device):
            output = model(emails)
            reconstruction_error = loss_fn(output, emails)

        recon_error_list.append(reconstruction_error.item())
        labels_list.append(label.item())



# Get the Threshold for recon_errors
threshold = np.percentile(recon_error_list, 25)

# Use the reconstruction error and threshold to determine ham vs spam
recon_error_list = np.array(recon_error_list)
predicted_classifications = (recon_error_list > threshold).astype(int)

# Calculate various eval metrics
precision = metrics.precision_score(labels_list, predicted_classifications)
recall = metrics.recall_score(labels_list, predicted_classifications)
f1_score = metrics.f1_score(labels_list, predicted_classifications)
roc_auc = metrics.roc_auc_score(labels_list, predicted_classifications)
conf_matrix = metrics.confusion_matrix(labels_list, predicted_classifications)

# Print out the testing stats
print('\n----- TESTING STATS -----')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1 Score:  {f1_score:.4f}')
print(f'ROC-AUC:   {roc_auc:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

torch.save([model.kwargs, model.state_dict()], 'model.pth')