import torch.nn as nn

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class PrunableNeuralModel(nn.Module):
    def __init__(self):
        super(PrunableNeuralModel, self).__init__()

        layer_dims = [784, 10, 10, 10]
        self.layers = nn.ModuleList()

        for i in range(1, len(layer_dims)):
            self.layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i], bias=False))
            if i != len(layer_dims) - 1:
                # Apply ReLU activation except in the last layer
                self.layers.append(nn.ReLU())
        # hook
        activation = {}

        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        def forward(self, x):
            h1 = self.layers[0].register_forward_hook(getActivation('layers[0]'))
            h2 = self.layers[2].register_forward_hook(getActivation('layers[2]'))
            h3 = self.layers[4].register_forward_hook(getActivation('layers[4]'))

            for layer in self.layers:
                x = layer(x)

            # print(activation)
            # for key, item in activation.items():
            #     print(key, item.size(dim=0))

            h1.remove()
            h2.remove()
            h3.remove()
            return x

def get_model(device):
    # Initialize the model using the custom architecture and move to the selected device
    model = PrunableNeuralModel().to(device)
    return model

def prune_model(model, rankings):
    return model
def reinit_model(model, weights):
    return model