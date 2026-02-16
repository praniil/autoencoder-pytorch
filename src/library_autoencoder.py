"""
Library-based Autoencoder using PyTorch's Sequential API
This implementation uses PyTorch's built-in Sequential container
which is more high-level and library-oriented compared to custom nn.Module
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


class SequentialAutoencoder(nn.Module):
    """
    Autoencoder using PyTorch's Sequential API
    This is more library-based as it uses Sequential containers
    """
    def __init__(self, input_dim=784, encoding_dim=128):
        super(SequentialAutoencoder, self).__init__()
        
        # Encoder using Sequential
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder using Sequential
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_library_autoencoder(epochs=100, batch_size=128, learning_rate=1e-3, patience=3):
    """
    Train the library-based autoencoder with the same hyperparameters
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transform
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load dataset
    full_dataset = torchvision.datasets.MNIST(
        root="../mnist_dataset", train=True, transform=transform, download=True
    )
    
    # Split dataset
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - (train_size + val_size)
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # Initialize model
    model = SequentialAutoencoder(input_dim=784, encoding_dim=128).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Early stopping
    best_loss = None
    no_improvement_count = 0
    
    # Training history
    train_losses = []
    val_losses = []
    
    print("\nTraining Library-based Autoencoder...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        for batch_features, _ in train_loader:
            batch_features = batch_features.view(-1, 784).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = criterion(outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            
            epoch_train_loss += train_loss.item()
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch_features, _ in val_loader:
                batch_features = batch_features.view(-1, 784).to(device)
                outputs = model(batch_features)
                val_loss = criterion(outputs, batch_features)
                epoch_val_loss += val_loss.item()
        
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.8f}, Val Loss: {epoch_val_loss:.8f}")
        
        # Early stopping check
        if best_loss is None or epoch_val_loss < best_loss - 1e-5:
            best_loss = epoch_val_loss
            no_improvement_count = 0
            # Save best model
            torch.save(model.state_dict(), "../trained-library-autoencoder")
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} epochs")
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("=" * 60)
    print("Training complete!\n")
    
    return model, train_losses, val_losses, test_dataset


if __name__ == "__main__":
    model, train_losses, val_losses, test_dataset = train_library_autoencoder()
    print(f"Final model saved to: ../trained-library-autoencoder")
    print(f"Best validation loss: {min(val_losses):.8f}")
