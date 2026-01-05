import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# hyper-params
batch_size = 512
epochs = 20
learning_rate = 1e-5
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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    dataset= test_dataset, batch_size= 32, shuffle= False, num_workers=4
)

# training the network
for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:  #labels are ignored _ used
        batch_features = batch_features.view(-1, 784).to(device)

        #reset the gradients back to zero
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features)

        # compute loss
        train_loss = craiterion(batch_features, outputs)

        # compute accumulated gradients
        train_loss.backward()
        
        # parameter update based on current gradients
        optimizer.step()

        # mini-batch training loss and epoch loss added
        loss += train_loss.item()
    
    #compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))    