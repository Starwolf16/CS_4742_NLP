import string
import numpy as np
import os

print(os.getcwd())


with open('Assignment_1\sample_txt.txt') as content:
    text = content.read()

    text = text.strip('\n')
    text = text.translate(str.maketrans('','',string.punctuation))

    text = np.array(text.split())
    print(text)
    print(len(np.unique(text)))