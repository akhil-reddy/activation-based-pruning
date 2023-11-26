import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.utils as vutils

ff_layers = [
    nn.Linear(64 * 16 * 16, 128 * 4 * 4),
    nn.Linear(128 * 4 * 4, 256 * 4 * 4),
    nn.Linear(256 * 4 * 4, 100),
    nn.Linear(100, 256 * 4 * 4),
    nn.Linear(256 * 4 * 4, 128 * 4 * 4),
    nn.Linear(128 * 4 * 4, 64 * 16 * 16)
]

# Define a simple VAE model
class VAE(nn.Module):
    def __init__(self, ff_layers):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            ff_layers[0],
            nn.ReLU(),
            ff_layers[1],
            nn.ReLU(),
        )

        self.fc_mu = ff_layers[2]
        self.fc_logvar = ff_layers[2]

        self.decoder = nn.Sequential(
            ff_layers[3],
            nn.ReLU(),
            ff_layers[4],
            nn.ReLU(),
            ff_layers[5],
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

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

    def reinit_model(self, weights, layers, device, input_layer, skip_layer):
        layer_dims = [input_layer]
        for layer in layers:
            layer_dims.append(len([i for i in layer if i == 1]))

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
                    ff_layers[i].weight = nn.Parameter(torch.FloatTensor(new_weights[i]))

        return VAE(ff_layers).to(device)

# Define the loss function for VAE
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Set up data transformation and loader
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = datasets.CelebA(root='data/celeba', split='all', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# Initialize the VAE model and optimizer
vae = VAE(ff_layers)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, data in enumerate(dataloader):
        img, _ = data
        img = Variable(img)

        optimizer.zero_grad()

        recon_batch, mu, logvar = vae(img)
        loss = loss_function(recon_batch, img, mu, logvar)

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, batch_idx+1, len(dataloader), loss.item()
            ))

            # Visualize original and reconstructed images
            if batch_idx % 500 == 0:
                with torch.no_grad():
                    vae.eval()
                    sample = img[:4]
                    recon_sample, _, _ = vae(sample)

                    comparison = torch.cat([sample, recon_sample])
                    comparison = comparison.view(-1, 3, 64, 64)
                    comparison = vutils.make_grid(comparison, nrow=4, padding=5, normalize=True)

                    plt.imshow(comparison.permute(1, 2, 0).numpy())
                    plt.show()

                vae.train()

# Save the trained model
torch.save(vae.state_dict(), 'vae_celeba.pth')