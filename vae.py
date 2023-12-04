import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from ranking import *

ff_layers = [
    nn.Linear(64 * 16 * 16, 128 * 4 * 4, bias=False),
    nn.Linear(128 * 4 * 4, 256 * 4 * 4, bias=False),
    nn.Linear(100, 256 * 4 * 4, bias=False),
    nn.Linear(256 * 4 * 4, 128 * 4 * 4, bias=False),
    nn.Linear(128 * 4 * 4, 64 * 16 * 16, bias=False)
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

        self.fc_mu = nn.Linear(256 * 4 * 4, 100, bias=False)
        self.fc_logvar = nn.Linear(256 * 4 * 4, 100, bias=False)

        self.decoder = nn.Sequential(
            ff_layers[2],
            nn.ReLU(),
            ff_layers[3],
            nn.ReLU(),
            ff_layers[4],
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
    weights are all weights for FF layers in encoder AND decoder, in sequence 
    layers are the 1/-1 coded FF layers in encoder AND decoder, in sequence
    mu_weight is the weight matrix of the mu layer
    logvar_weight is the weight matrix of the logvar layer
    '''

    def reinit_model(self, weights, layers, device):
        for i in range(len(weights)):
            weights[i] = torch.Tensor.tolist(weights[i])

        new_weights = []
        ff_weights = weights[4:7] + weights[8:11]
        mu_weight = weights[6]
        logvar_weight = weights[7]

        # Format new weights
        for i in range(len(layers)):
            layer_weights = []
            curr_layer_pruned_neurons = []
            for j in range(len(layers[i])):
                neuron_weights = ff_weights[i][j]
                # Neuron to prune
                if layers[i][j] == -1:
                    curr_layer_pruned_neurons.append(j)
                else:
                    layer_weights.append(neuron_weights)
            new_weights.append(layer_weights)

            if i + 1 != len(layers):
                # Live neurons
                live = [x for x in list(range(len(layers[i]))) if x not in curr_layer_pruned_neurons]
                for k in range(len(layers[i + 1])):
                    # Special case of mu & sigma
                    if i + 1 == 2:
                        mu_outflow = []
                        logvar_outflow = []
                        for j in live:
                            mu_outflow.append(mu_weight[k][j])
                            logvar_outflow.append(logvar_weight[k][j])
                        mu_weight[k] = mu_outflow
                        logvar_weight[k] = logvar_outflow
                    else:
                        # Only weights from live neurons in the current layer
                        live_outflow = []
                        for j in live:
                            live_outflow.append(ff_weights[i + 1][k][j])
                        ff_weights[i + 1][k] = live_outflow

        del layers[2]
        del new_weights[2]
        # Reinitialize new weights to the network
        with torch.no_grad():
            for i in range(len(layers)):
                    ff_layers[i].weight = nn.Parameter(torch.FloatTensor(new_weights[i]))

        model = VAE(ff_layers).to(device)

        model.encoder[0].weight = nn.Parameter(torch.FloatTensor(weights[0]))
        model.encoder[0].bias = nn.Parameter(torch.FloatTensor(weights[1]))
        model.encoder[2].weight = nn.Parameter(torch.FloatTensor(weights[2]))
        model.encoder[2].bias = nn.Parameter(torch.FloatTensor(weights[3]))

        model.fc_mu.weight = nn.Parameter(torch.FloatTensor(mu_weight))
        model.fc_logvar.weight = nn.Parameter(torch.FloatTensor(logvar_weight))

        model.decoder[7].weight = nn.Parameter(torch.FloatTensor(weights[11]))
        model.decoder[7].bias = nn.Parameter(torch.FloatTensor(weights[12]))
        model.decoder[9].weight = nn.Parameter(torch.FloatTensor(weights[13]))
        model.decoder[9].bias = nn.Parameter(torch.FloatTensor(weights[14]))

        return model

# Define the loss function for VAE
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(dataloader,vae,optimizer,increment=10):
    num_epochs = increment
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


if __name__ == "__main__":   
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

    #train(dataloader,vae,optimizer)

    # Save the trained model
    torch.save(vae.state_dict(), 'vae_celeba.pth')

    total_epochs = 10
    initial_iterations = 5
    increment = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weightMatrix = {}
    for name, param in vae.state_dict().items():
        weightMatrix[name] = param

    for i in range(initial_iterations + 1, total_epochs + 1, increment):
        # Perform pruning and retraining
        rankings, max_ranking = getRandomScoresVAE(weightMatrix)
        layers = vae.prune_model_from_rankings(rankings, max_ranking)
        vae = vae.reinit_model(list(weightMatrix.values()), layers, device)
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)
        
        train(dataloader,vae,optimizer,increment=increment)
      
        # Update the weight matrix for the next iteration
        for name, param in vae.state_dict().items():
            weightMatrix[name] = param