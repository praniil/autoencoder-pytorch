import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], 
            out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, 
            out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128,
            out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128,
            out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        encode = self.encoder_output_layer(activation)
        encode = torch.relu(encode)
        activation = self.decoder_hidden_layer(encode)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed