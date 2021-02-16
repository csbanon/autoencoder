import numpy as np
import matplotlib.pyplot as plt

def train(model, device, train_loader, criterion, optimizer):
    """
    Trains the model and optimizes it.

    :model: the model to train.
    :device: the device used for training ('cuda' or 'cpu').
    :train_loader: the DataLoader for training samples.
    :criterion: the loss function.
    :optimizer: the optimizer function.
    """

    print("Training the autoencoder:")

    # Hyperparameters
    learning_rate = 0.003
    num_epochs = 10
    batch_size = 128

    loss_list = np.zeros(num_epochs)

    for epoch in range(num_epochs):

        for batch in train_loader:

            # Set up the images.
            images, _ = batch
            images = images.to(device)

            # Perform forward propagation.
            output = model(images)
            loss = criterion(output, images)

            # Perform backpropagation.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list[epoch] = loss

        # Print the loss.
        print('Epoch {}/{} - Loss: {:.5f}'.format(epoch + 1, num_epochs, loss))

    print("Finished training!")

    # Plot the training loss.
    fig = plt.figure()
    ax = plt.axes()
    x = np.arange(1, num_epochs + 1, 1)
    plt.title('Loss Per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(x, loss_list)
