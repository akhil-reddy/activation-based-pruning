from model import *
from ranking import *

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
activation_weight_values = {}
layer_dims = [784, 10, 10, 10]
model = PrunableNeuralModel(layer_dims).to(device)
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


total_epochs = 100
inital_iterations = 50
train(model, train_loader, dev_loader, optimizer, criterion, inital_iterations)

weightMatrix = {}
# to get weights
for name, param in model.state_dict().items():
    print(name, param)
    weightMatrix[name] = param

# to get activation values
print(model.activation_values)

increment = 5
for i in range(total_epochs-inital_iterations+1, total_epochs+1, increment):
    # call akhils function
    scores = getRandomScores(weightMatrix)
    pruned_model = model
    train(model, train_loader, dev_loader, optimizer, criterion, increment)
