import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from model import Autoencoder
from train import train

def run():
    batch_size = 128
    learning_rate = 0.003

    # Set up the GPU if available.
    use_cuda = torch.cuda.is_available() 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Torch Device Selected: ", device)

    # Initialize the model and send it to the GPU.
    model = Autoencoder().to(device)

    # Define the loss function and optimizer.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Create transformations to apply to each data sample.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Set up the training and test set.
    train_set = MNIST('./data/', train=True, download=True, transform=transform)
    test_set = MNIST('./data/', train=False, transform=transform)

    # Set up the DataLoader.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # Train the autoencoder.
    train(model, device, train_loader, criterion, optimizer)

if __name__ == '__main__':

    run()