parameter = {
    'total_epochs': 400,
    'inital_iterations': 5,
    'increment':50,

    'MNIST':{
        'normalize':[0.5],
        'batch_size': 41000,
        'layer_dims':[784, 50, 30, 10],
        'X_batch_view': [-1,784] # input dimensions
    },
    'FashionMNIST':{
        'normalize':[0.5],
        'batch_size': 41000,
        'layer_dims':[784, 50, 30, 10],
        'X_batch_view': [-1,784] # input dimensions
    },
    'CIFAR10':{
        'normalize':[0.5, 0.5, 0.5],
        'batch_size': 41000,
        'layer_dims':[3072, 50, 30, 10],
        'X_batch_view': [-1, 3072] # input dimensions
    }
    
}