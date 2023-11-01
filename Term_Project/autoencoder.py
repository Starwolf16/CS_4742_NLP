import torch
import torch.nn as nn
from math import floor

class AutoEncoder():
    def __init__(self, in_features, latent_size):
        self.in_feature = in_features
        self.in_feature_large = floor(self.in_feature * 0.75)
        self.in_feature_med = floor(self.in_feature * 0.5)
        self.in_feature_small = floor(self.in_feature * 0.25)

        self.latent_size = latent_size

        self.kwargs = {
            'In Features': self.in_feature, 
            'Latent_size': self.latent_size
        }

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=5, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=5, padding=1),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=5, stride=5, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 48, kernel_size=5, stride=5, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(48, 32, kernel_size=5, stride=5, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=5, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=5, padding=1),
            nn.ReLU(),
        )

        # self.fully_connected = nn.Sequential(
        #     nn.Linear(self.in_feature, self.in_feature_large),
        #     nn.ReLU(),
        #     nn.Linear(self.in_feature_large, self.in_feature_med),
        #     nn.ReLU(),
        #     nn.Linear(self.in_feature_med, self.in_feature_small),
        #     nn.ReLU(),
        #     nn.Linear(self.in_feature_small, self.latent_size),
        #     nn.ReLU(),
        #     nn.Linear(self.latent_size, self.in_feature_small),
        #     nn.ReLU(),
        #     nn.Linear(self.in_feature_small, self.in_feature_med),
        #     nn.ReLU(),
        #     nn.Linear(self.in_feature_med, self.in_feature_large),
        #     nn.ReLU(),
        #     nn.Linear(self.in_feature_large, self.in_feature),
        #     nn.ReLU(),
        # )

    
    def forward_pass(self, text, target):
        latent_rep = self.encoder(text)
        reconstructed = self.decoder(latent_rep)

        return reconstructed
    

