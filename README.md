# CSCI 567 - Machine Learning

## How to run the model? 

### Dependencies

-  Python 3.7+ Interpreter
-  Torch 1.13.1+
-  TorchVision 0.14.1+
-  NumPy 1.17.2
-  Matplotlib 3.1.1+

### Classification Tasks

- The entry point to running various classification models would be driver.py. 
- Running driver.py prompts the user to select the desired dataset -  1 for MNIST, 2 for FashionMNIST, 3 for CIFAR-10 and 4 for CIFAR-100.
- In constants.py, the hyperparameters are 'total_epochs': Total number of epochs , 'inital_iterations': Number of epochs to train the model for initial maturity and 'increment': Frequency at which train-time pruning needs to be invoked.
- In ranking.py, toggle the contribBasedPruning flag to true for contribution-based neuron pruning and false for randomised pruning

### Generative Tasks

Run vae.py with toggling between lines 223 and 224 for switching between randomised pruning and rank-based pruning.

## How to generate the plots?

Example for running a comparison between a feed forward network without pruning, with importance based pruning and with randomised pruning -

- Run the driver with 3 different configs:
  - Set 'total_epochs' as 100, 'inital_iterations' as 100 and 'increment' as 10 and run the driver. (Without any pruning) and log file enabled in driver.py is training_logwithoutPruning.txt
  - Now set 'total_epochs' as 100, 'inital_iterations' as 30 and 'increment' as 10, run the driver and make sure contribBasedPruning set as True (With importance based pruning) and the log file enabled in driver.py is training_logwithPruning.txt
  - Now set 'total_epochs' as 100, 'inital_iterations' as 30 and 'increment' as 10, run the driver and make sure contribBasedPruning set as False (With importance based pruning) and log file enabled in driver.py is training_logworstPruning.txt
- Run plot.py to generate the plots for viewing the pyplots for accuracy.

## How to download the datasets

Datasets are automatically downloaded onto your local Python virtual environment by Torch. As the files are executed, internal checks download the dataset for the first time. No explicit download needed. 
