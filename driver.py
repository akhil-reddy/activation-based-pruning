# Module Imports
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from constants import parameter 

# Local imports
from models.feedforward import prune_model_from_rankings, reinit_model
from ranking import *

# Import helper
from helper import switch, get_model

print('Choose a dataset from the below options \n 1. MNIST \n 2. FashionMNIST \n 3. CIFAR-10 \n 4. CIFAR-100')
print("Enter the value: ")
val = int(input())
dataset = switch(val)

torch.manual_seed(30)
random.seed(30)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(tuple(parameter[dataset]['normalize']),tuple(parameter[dataset]['normalize']))])
train_dataset = getattr(datasets, dataset)(root='./datasets', train=True, transform=transform, download=True)
# test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=parameter[dataset]['batch_size'], shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

layer_dims = parameter[dataset]['layer_dims']   # Adjust this based on your model architecture

model = get_model(layer_dims, device, dataset)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Modified Training Loop
def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch.view(parameter[dataset]['X_batch_view']))  # Adjust the input dimensions 
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            # Logging the progress
            if i % 10 == 0:
                with torch.no_grad():
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                    accuracy = (predictions == Y_batch.cpu().numpy()).mean()
                    print(f"Epoch: {epoch + 1}, Batch: {i}, Loss: {loss.item()}, Accuracy: {accuracy * 100}%")

total_epochs = parameter['total_epochs']
inital_iterations = parameter['inital_iterations']
train(model, train_loader, optimizer, criterion, inital_iterations)

weightMatrix = {}
for name, param in model.state_dict().items():
    weightMatrix[name] = param

increment = parameter['increment']
for i in range(inital_iterations + 1, total_epochs + 1, increment):
    rankings, max_ranking = getLocalRanks(weightMatrix, model.activation_values)
    #rankings, max_ranking = getRandomScores(weightMatrix)
    layers = prune_model_from_rankings(rankings, max_ranking)
    model = reinit_model(list(weightMatrix.values()), layers, device, layer_dims[0])
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    train(model, train_loader, optimizer, criterion, increment)

    for name, param in model.state_dict().items():
        weightMatrix[name] = param
        


""" In case we want to use a url instead of pytorch datasets

from model import *
from models.basic_feedforward import *
from ranking import *

import os
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import pandas as pd
from sklearn.utils import shuffle
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

torch.manual_seed(30)
random.seed(30)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()
url = os.environ['URL']
data = pd.read_csv(url)

print("Dataset Info:")
print(data.info())

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Summary statistics of the dataset
print("\nSummary Statistics:")
print(data.describe())

# Number of rows and columns
print("\nNumber of rows and columns:")
print(data.shape)

# Column names
print("\nColumn Names:")
print(data.columns)

# Data types of columns
print("\nData Types:")
print(data.dtypes)

# Missing values
print("\nMissing Values:")
print(data.isnull().sum())

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
layer_dims = [784, 50, 30, 10]
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
            if i / 10 == 0:
                with torch.no_grad():
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                    accuracy = (predictions == Y_batch.cpu().numpy()).mean()
                    print(f"Epoch: {epoch + 1}, Batch: {i}, Loss: {loss.item()}, Accuracy: {accuracy * 100}%")


total_epochs = 400
inital_iterations = 200
train(model, train_loader, dev_loader, optimizer, criterion, inital_iterations)

weightMatrix = {}
# to get weights
for name, param in model.state_dict().items():
    #print(name, param)
    weightMatrix[name] = param

# to get activation values
#print(model.activation_values)

increment = 50
for i in range(inital_iterations + 1, total_epochs + 1, increment):
    rankings, max_ranking = getLocalRanks(weightMatrix, model.activation_values)
    #rankings, max_ranking = getRandomScores(weightMatrix)
    layers = prune_model_from_rankings(rankings, max_ranking)
    model = reinit_model(list(weightMatrix.values()), layers, device, layer_dims[0])
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    train(model, train_loader, dev_loader, optimizer, criterion, increment)

    for name, param in model.state_dict().items():
        #print(name, param)
        weightMatrix[name] = param

"""