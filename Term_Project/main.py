import matplotlib.pyplot as plt
import pandas as pd
import torch

from spacy.lang.en import English
from sys import exit
from torch.cuda.amp import GradScaler

nlp = English()

dataset = pd.read_csv("emails.csv")
print(dataset.columns)

CORPUS = dataset.pop("text")
print(CORPUS)
