import torch
import torch.nn as nn
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class CNN(nn.Module):
    def __init__(self, layer_dims, conv_dims):
        super(CNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()  
        self.activation_values = {}

        
        for conv_dim in conv_dims:
            in_channels, out_channels, kernel_size, stride, padding = conv_dim
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.flatten = nn.Flatten()

        # Add fully connected layers
        for i in range(1, len(layer_dims)):
            self.fc_layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i], bias=False))
            if i != len(layer_dims) - 1:
                # Apply ReLU activation except in the last layer
                self.fc_layers.append(nn.ReLU())

        # hook
        self.activation = {}
    def getActivation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def forward(self, x):
       
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

      
        x = self.flatten(x)

      
        h1 = self.fc_layers[0].register_forward_hook(self.getActivation('fc_layers[0]'))
        h2 = self.fc_layers[2].register_forward_hook(self.getActivation('fc_layers[2]'))
        h3 = self.fc_layers[4].register_forward_hook(self.getActivation('fc_layers[4]'))

       
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

       
        for key, item in self.activation.items():
            self.activation_values[key] = item

        
        h1.remove()
        h2.remove()
        h3.remove()

        return x

    def prune_model_from_rankings(self, rankings, max_ranking, prune_percent=10):
        layers = []
        for layer_index in range(len(rankings)):
            layer = []
            for neuron_rank in rankings[layer_index]:
                if neuron_rank > max(1, (1 - 0.01 * prune_percent) * max_ranking[layer_index]):
                    layer.append(-1)
                else:
                    layer.append(1)
            layers.append(layer)
        return layers


    '''
    weights are from layer (0 to n] with n-1 elements
    layers are from [1 to n] with n-1 elements
    '''


    def reinit_model(self, weights, layers, device, input_layer, conv_dims):
        layer_dims = [input_layer]
        for layer in layers:
            layer_dims.append(len([i for i in layer if i == 1]))

        model = CNN(layer_dims, conv_dims).to(device)
        for i in range(len(weights)):
            weights[i] = torch.Tensor.tolist(weights[i])

        with torch.no_grad():
            for i in range((len(weights)- len(layers))//2):
                model.conv_layers[3 * i].weight = nn.Parameter(torch.FloatTensor(weights[2*i]))    
                model.conv_layers[3 * i].bias = nn.Parameter(torch.FloatTensor(weights[2*i + 1]))        
        
        weights = weights[2*i+2:]

        new_weights = []
        # Format new weights
        for i in range(len(layers)):
            layer_weights = []
            curr_layer_pruned_neurons = []
            for j in range(len(layers[i])):
                neuron_weights = weights[i][j]
                # Neuron to prune
                if layers[i][j] == -1:
                    if i + 1 != len(layers):
                        curr_layer_pruned_neurons.append(j)
                else:
                    layer_weights.append(neuron_weights)
            new_weights.append(layer_weights)

            if i + 1 != len(layers):
                # Live neurons
                live = [x for x in list(range(len(layers[i]))) if x not in curr_layer_pruned_neurons]
                for k in range(len(layers[i + 1])):
                    live_outflow = []
                    # Only weights from live neurons in the current layer
                    for j in live:
                        live_outflow.append(weights[i + 1][k][j])
                    weights[i + 1][k] = live_outflow

        # Reinitialize new weights to the network
        with torch.no_grad():
            for i in range(len(layers)):
                model.fc_layers[2 * i].weight = nn.Parameter(torch.FloatTensor(new_weights[i]))

        return model