parameter = {
    'total_epochs': 100,
    'inital_iterations': 5,
    'increment':5,

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
    'normalize': [0.5, 0.5, 0.5], 
    'batch_size': 128,  
    'conv_dims': [(3, 16, 3, 1, 1), (16, 32, 3, 1, 1)],
    'layer_dims': [32 * 8 * 8, 256, 128, 10],
    'X_batch_view': [-1, 3, 32, 32] 
    },
     'CIFAR100':{
     'normalize': [0.5, 0.5, 0.5],  
    'batch_size': 128, 
    'conv_dims': [(3, 16, 3, 1, 1), (16, 32, 3, 1, 1)], 
    'layer_dims': [32 * 8 * 8, 256, 128, 100],
    'X_batch_view': [-1, 3, 32, 32] 
    }
    # 46.36 - 49.1
}
# CIFAR 100 and Collab
# Graph of training accuray vs no of epochs - random pruning, our pruning (our should have shallower)