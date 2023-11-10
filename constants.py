parameter = {
    'total_epochs': 1000,
    'inital_iterations': 600,
    'increment':100,

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
        'layer_dims':[3072, 512, 256, 10], # 60.64
        'X_batch_view': [-1, 3072] # input dimensions
    },
     'CIFAR100':{
        'normalize':[0.5, 0.5, 0.5],
        'batch_size': 41000,
        'layer_dims':[3072, 512, 256, 128, 100], # 22.37
        'X_batch_view': [-1, 3072] # input dimensions
    }
    # 46.36 - 49.1
}
# CIFAR 100 and Collab
# Graph of training accuray vs no of epochs - random pruning, our pruning (our should have shallower)