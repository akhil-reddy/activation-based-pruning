import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import pandas as pd
from sklearn.utils import shuffle

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Set random seed for reproducibility
torch.manual_seed(30)
random.seed(30)

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
url = 'https://raw.githubusercontent.com/BudhiaRishabh/CSCI-567-ML/main/train.csv'
data = pd.read_csv(url)

data = shuffle(data, random_state=30)

data_dev = data.iloc[:1000].values
Y_dev = data_dev[:, 0]
X_dev = data_dev[:, 1:] / 255.

data_train = data.iloc[1000:].values
Y_train = data_train[:, 0]
X_train = data_train[:, 1:] / 255.

# Convert data to PyTorch tensors and move them to the selected device
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
Y_train = torch.tensor(Y_train, dtype=torch.long, device=device)
X_dev = torch.tensor(X_dev, dtype=torch.float32, device=device)
Y_dev = torch.tensor(Y_dev, dtype=torch.long, device=device)

# Define a customizable neural network class
class CustomNeuralNetwork(nn.Module):
    def __init__(self, layer_dims):
        super(CustomNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(layer_dims)):
            self.layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            if i != len(layer_dims) - 1:
                self.layers.append(nn.ReLU())  # Apply ReLU activation except in the last layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Define the architecture of the neural network (customize as needed)
layer_dims = [784, 10, 10, 10]  # Example: 3 layers with 10 neurons each

# Initialize the model using the custom architecture and move to the selected device
model = CustomNeuralNetwork(layer_dims).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
def train(model, X, Y, optimizer, criterion, iterations):
    model.train()
    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                accuracy = (predictions == Y.cpu().numpy()).mean()
                print(f"Iteration: {i}, Loss: {loss.item()}, Accuracy: {accuracy * 100}%")

st = time.time()
train(model, X_train, Y_train, optimizer, criterion, 500)
et = time.time()

print(et - st)