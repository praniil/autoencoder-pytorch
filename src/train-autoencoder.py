from autoencoder import AutoEncoder
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import random_split

class EarlyStopping:
    def __init__(self, patience=5, delta = 0, verbose = False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            print(f"no improvements for {self.no_improvement_count}")
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no imporovement has been observed")


if __name__ == "__main__": 
    # hyper-params
    batch_size = 128
    epochs = 100
    learning_rate = 1e-3

    transform = transforms.Compose([transforms.ToTensor()])

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

    full_dataset = torchvision.datasets.MNIST(
        root="../mnist_dataset", train=True, transform=transform, download=True
    )

    # ratio of split
    train_ratio = 0.8
    val_ratio = 0.2

    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size - train_size)

    # perform the split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # data loader

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # init early stopping
    early_stopping = EarlyStopping(patience=3, delta=1e-5, verbose=True)

    # training the network
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        #training phase
        model.train()
        for batch_features, _ in train_loader: 
         #labels are ignored _ used
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
            epoch_train_loss += train_loss.item()
        
        #compute the epoch training loss
        epoch_train_loss = epoch_train_loss / len(train_loader)

        # validation phase
        model.eval()
        with torch.no_grad():
            for batch_features, _ in val_loader:  #labels are ignored _ used
                batch_features = batch_features.view(-1, 784).to(device)

                # compute reconstructions
                outputs = model(batch_features)

                # compute loss
                val_loss = craiterion(batch_features, outputs)
                
                # mini-batch training loss and epoch loss added
                epoch_val_loss += val_loss.item()

        epoch_val_loss = epoch_val_loss / len(val_loader)

        # display the epoch training loss
        print("epoch : {}/{}, train loss = {:.8f}".format(epoch + 1, epochs, epoch_train_loss))   
        print("epoch : {}/{}, val loss = {:.8f}".format(epoch + 1, epochs, epoch_val_loss))   

        early_stopping.check_early_stop(val_loss=epoch_val_loss)

        if early_stopping.stop_training:
            print(f"Early stopiing at epoch {epoch}")
            break

    # save the model
    path = "../trained-autoencoder" 

    torch.save(model.state_dict(), path)