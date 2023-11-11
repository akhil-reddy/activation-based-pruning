from models.cnn import CNN
from models.feedforward import FeedForward
from models.vae import VAE


def get_model(layer_dims, device, dataset,conv_dims=None):
    # Initialize the model using the custom architecture and move to the selected device
    if dataset == "MNIST" or dataset == "FashionMNIST" :
        return FeedForward(layer_dims).to(device)
    elif dataset == "CIFAR10" or dataset == "CIFAR100":
        return CNN(layer_dims,conv_dims).to(device)
    elif dataset == "VAE" :
       return VAE(layer_dims).to(device)

    return None

def switch(num):
    if num == 1:
        return "MNIST"
    elif num == 2:
        return "FashionMNIST"
    elif num == 3:
        return "CIFAR10"
    elif num == 4:
        return "CIFAR100"

