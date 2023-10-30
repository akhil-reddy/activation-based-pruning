import torch
import torch.nn as nn

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class PrunableNeuralModel(nn.Module):
    def __init__(self, layer_dims):
        super(PrunableNeuralModel, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(1, len(layer_dims)):
            self.layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i], bias=False))
            if i != len(layer_dims) - 1:
                # Apply ReLU activation except in the last layer
                self.layers.append(nn.ReLU())
        # hook
        self.activation = {}

    def getActivation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def forward(self, x):
        h1 = self.layers[0].register_forward_hook(self.getActivation('layers[0]'))
        h2 = self.layers[2].register_forward_hook(self.getActivation('layers[2]'))
        h3 = self.layers[4].register_forward_hook(self.getActivation('layers[4]'))

        for layer in self.layers:
            x = layer(x)

        # print(activation)
        # for key, item in activation.items():
        #     print(key, item.size(dim=0))

        h1.remove()
        h2.remove()
        h3.remove()
        return x


def get_model(layer_dims, device):
    # Initialize the model using the custom architecture and move to the selected device
    model = PrunableNeuralModel(layer_dims).to(device)
    return model


def prune_model_from_rankings(rankings, max_ranking, prune_percent):
    layers = []
    for layerwise_rank in rankings:
        layer = []
        for neuron_rank in layerwise_rank:
            if neuron_rank > (1 - 0.01 * prune_percent) * max_ranking:
                layer.append(-1)
            else:
                layer.append(1)
        layers.append(layer)
    return layers


'''

weights are from layer (0 to n] with n-1 elements
layers are from [1 to n] with n-1 elements
'''


def reinit_model(weights, layers, device):
    layer_dims = []
    for layer in layers:
        layer_dims.append(len([i for i in layer if i == 1]))

    model = get_model(layer_dims, device)

    # Format new weights
    for i in range(len(layers)):
        for j in range(len(layers[i])):
            # Neuron to prune
            if layers[i][j] == -1:
                # Remove weights from inflow edges, i.e., weights[i][j]
                del weights[i][j]
                if i + 1 != len(layers):
                    # Remove weights from outflow edges, i.e., weights[i+1][][j]
                    for k in range(len(layers[i + 1])):
                        del weights[i + 1][k][j]

    # Reinitialize new weights to the network
    with torch.no_grad():
        for i in range(len(layers)):
            model.layers[i].weight = torch.Tensor(weights[i])

    return model