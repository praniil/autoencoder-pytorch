from autoencoder import AutoEncoder
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import train_autoencoder as ta


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([transforms.ToTensor()])

loaded_model = AutoEncoder(input_shape = 784).to(device)
loaded_model.load_state_dict(torch.load("../trained-autoencoder", map_location=device))
loaded_model.eval()

test_dataset = ta.test_dataset

test_loader = torch.utils.data.DataLoader(
    dataset= test_dataset, batch_size= 32, shuffle= False, num_workers=4
)


# get one batch
with torch.no_grad():
    batch_features, _ = next(iter(test_loader))
    batch_features = batch_features.view(-1, 784).to(device)

    originals = batch_features[10:20]
    reconstructions = loaded_model(originals)

# plotting
plt.figure(figsize=(20, 4))

for i in range(10):
    # original
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(originals[i].cpu().numpy().reshape(28, 28), cmap="gray")
    ax.axis("off")

    # reconstructed
    ax = plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructions[i].cpu().numpy().reshape(28, 28), cmap="gray")
    ax.axis("off")

plt.show()
