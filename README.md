# CSCI 567 - Machine Learning

## How to run the model? 

### Classification Tasks

- The entry point to running various CNN models would be driver.py. 
- Running driver.py prompts the user to select the desired model  1 for MNIST 2 for FashionMNIST,  3 for CIFAR-10 or 4 for CIFAR-100.
- In constants.py 'total_epochs': Total number of epochs , 'inital_iterations': Number of epochs to train the model for some initial maturity and 'increment': Frequency at which train time pruning needs to be invoked.
- In ranking.py toggle the contribBasedPruning flag to operate in contribution based neuron pruning (set to True) and randomised pruning (set to False)

Example for running a comparison between running a feed forward network without pruning, with importance based pruning and randomised pruning -

- Run the driver with 3 diff configs:
  - First set 'total_epochs' as 100, 'inital_iterations' as 100 and 'increment' as 10 and run the driver. (Without pruning) and log file enabled in driver.py is training_logwithoutPruning.txt
  - Second set 'total_epochs' as 100, 'inital_iterations' as 30 and 'increment' as 10 and run the driver and make sure contribBasedPruning set as True (With importance based pruning) and log file enabled in driver.py is training_logwithPruning.txt
  - Third set 'total_epochs' as 100, 'inital_iterations' as 30 and 'increment' as 10 and run the driver and make sure contribBasedPruning set as False (With importance based pruning) and log file enabled in driver.py is training_logworstPruning.txt
- Run plot.py to generate the plots for viewing the trends of accuracy.

### Generative Tasks

Run vae.py toggling between lines 223 and 224 for switching between randomised pruning and rank based pruning.


