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

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoEncoder(input_shape=784)
model = model.to(device=device)


# define an optimizer
# Adam optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# define a costfunction
craiterion = nn.MSELoss()

# transform
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="../mnist_dataset", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="../mnist_dataset", train=False, transform= transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    dataset= test_dataset, batch_size= 32, shuffle= False, num_workers=4
)
