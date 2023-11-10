from models.cnn import CNN
from models.feedforward import FeedForward
from models.vae import VAE


def get_model(layer_dims, device, type):
    # Initialize the model using the custom architecture and move to the selected device
    if type == 'ff':
        model = FeedForward(layer_dims).to(device)
    else:
        if type == 'cnn':
            model = CNN(layer_dims).to(device)
        else:
            model = VAE(layer_dims).to(device)
    return model
