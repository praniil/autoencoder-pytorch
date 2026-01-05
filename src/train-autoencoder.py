from autoencoder import AutoEncoder
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


if __name__ == "__main__": 
    # hyper-params
    batch_size = 512
    epochs = 20
    learning_rate = 1e-5

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

    train_dataset = torchvision.datasets.MNIST(
        root="../mnist_dataset", train=True, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
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

    # save the model
    path = "../trained-autoencoder" 

    torch.save(model.state_dict(), path)