import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Define a simple VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, 100)
        self.fc_logvar = nn.Linear(256 * 4 * 4, 100)

        self.decoder = nn.Sequential(
            nn.Linear(100, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
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

dataset = datasets.CelebA(root='path/to/celeba', split='all', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# Initialize the VAE model and optimizer
vae = VAE()
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