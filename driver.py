from examples.gradientDescent import *

model = CustomNeuralNetwork(layer_dims).to(device)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

total_epochs = 100
inital_iterations = 50
train(model, train_loader, dev_loader, optimizer, criterion, inital_iterations)

# to get weights
for name, param in model.state_dict().items():
    print(name, param)    

print("------------")

# to get activation values
print(activation_weight_values)

increment = 5
for i in range(total_epochs-inital_iterations+1, total_epochs+1, increment):
    # call akhils function
    pruned_model = model
    train(model, train_loader, dev_loader, optimizer, criterion, increment)
