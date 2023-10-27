import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import time
import pandas as pd
from sklearn.utils import shuffle
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

torch.manual_seed(30)
random.seed(30)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

url = 'https://raw.githubusercontent.com/BudhiaRishabh/CSCI-567-ML/main/train.csv'
data = pd.read_csv(url)
data = shuffle(data, random_state=30)

data_dev = data.iloc[:1000].values
Y_dev = data_dev[:, 0]
X_dev = data_dev[:, 1:] / 255.

data_train = data.iloc[1000:].values
Y_train = data_train[:, 0]
X_train = data_train[:, 1:] / 255.

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_dev = torch.tensor(X_dev, dtype=torch.float32)
Y_dev = torch.tensor(Y_dev, dtype=torch.long)

activation = {}

def getActivation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class CustomNeuralNetwork(nn.Module):
    def __init__(self, layer_dims):
        super(CustomNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(layer_dims)):
            self.layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i],bias=False))
            if i != len(layer_dims) - 1:
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        global activation
        activation = {}
        h1 = self.layers[0].register_forward_hook(getActivation('layers[0]'))
        h2 = self.layers[2].register_forward_hook(getActivation('layers[2]'))
        h3 = self.layers[4].register_forward_hook(getActivation('layers[4]'))
        for layer in self.layers:
            x = layer(x)
        h1.remove()
        h2.remove()
        h3.remove()
        return x

layer_dims = [784, 10, 10, 10]
model = CustomNeuralNetwork(layer_dims).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Convert data into batches using DataLoader
batch_size = 41000
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataset = TensorDataset(X_dev, Y_dev)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

# Modified Training Loop
def train(model, train_loader, dev_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        #print(activation)
        for i, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            
            
            # Logging the progress
            if i % 10 == 0:
                with torch.no_grad():
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                    accuracy = (predictions == Y_batch.cpu().numpy()).mean()
                    print(f"Epoch: {epoch+1}, Batch: {i}, Loss: {loss.item()}, Accuracy: {accuracy * 100}%")

st = time.time()
train(model, train_loader, dev_loader, optimizer, criterion, 5000)
et = time.time()

print(et - st)
