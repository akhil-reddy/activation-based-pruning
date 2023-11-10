import torch
import torch.nn as nn
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def prune_model_from_rankings(self,rankings, max_ranking, prune_percent=10):
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

    def reinit_model(self, weights, layers, device, input_layer):
        layer_dims = [input_layer]
        for layer in layers:
            layer_dims.append(len([i for i in layer if i == 1]))

        model = VAE(layer_dims).to(device)
        for i in range(len(weights)):
            weights[i] = torch.Tensor.tolist(weights[i])

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
                model.layers[2 * i].weight = nn.Parameter(torch.FloatTensor(new_weights[i]))

        return model
