# Module Imports
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from constants import parameter 

# Local imports
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
conv_dims = parameter[dataset]['conv_dims'] 
model = get_model(layer_dims, device, dataset, conv_dims)
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
    layers = model.prune_model_from_rankings(rankings, max_ranking)
    model = model.reinit_model(list(weightMatrix.values()), layers, device, layer_dims[0])
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    train(model, train_loader, optimizer, criterion, increment)

    for name, param in model.state_dict().items():
        weightMatrix[name] = param
    