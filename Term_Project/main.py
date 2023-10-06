import spacy 
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from sys import exit
from torch.cuda.amp import GradScaler

dataset = pd.read_csv("emails.csv")

