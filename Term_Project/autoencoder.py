import torch
import torch.nn as nn
from math import floor

class AutoEncoder(nn.Module):
    def __init__(self, in_features, latent_size):
        super().__init__()
        self.in_features = in_features
        self.in_feature_large = floor(self.in_features * 0.75)
        self.in_feature_med = floor(self.in_features * 0.5)
        self.in_feature_small = floor(self.in_features * 0.25)

        self.latent_size = latent_size

        self.kwargs = {
            'in_features': self.in_features, 
            'latent_size': self.latent_size
        }

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=10, stride=5),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 1, kernel_size=14, stride=5),
            nn.ReLU(),
        )


    
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        text = text.view(-1, 1, text.size(-1))
        latent_rep = self.encoder(text)
        reconstructed = self.decoder(latent_rep)
        reconstructed = reconstructed.squeeze(dim = 1)

        return reconstructed
    

