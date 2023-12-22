# Neural Importance-based Train-time Pruning

`Note: The project is structured for rapid experimentation and inference, and therefore might be less readable at times. 
Please refer to the report for a structured reading.`

## How to download the datasets?

Datasets are implicitly downloaded onto your local Python virtual environment by Torch. As the files are executed, internal checks download the dataset for the very first time only. Explicit downloads aren't required. 

## How to run the model? 

### Dependencies

-  Python 3.7+ Interpreter
-  Torch 1.13.1+
-  TorchVision 0.14.1+
-  NumPy 1.17.2
-  Matplotlib 3.1.1+

### Classification Tasks

- The entry point to running various classification models would be _driver.py_ 
- Running _driver.py_ prompts the user to select the desired dataset -  1 for MNIST, 2 for FashionMNIST, 3 for CIFAR-10 and 4 for CIFAR-100
- In _constants.py_, the hyperparameters are _total_epochs_: Total number of epochs , _inital_iterations_: Number of epochs to train the model for initial maturity and 'increment': Frequency at which train-time pruning needs to be invoked
- In _ranking.py_, toggle the _contribBasedPruning_ flag to true for contribution-based neuron pruning and false for randomised pruning

### Generative Tasks

Run _vae.py_ with toggling between lines 223 and 224 for switching between randomised pruning and rank-based pruning.

## How to generate the comparison plots for the classification tasks?

Example for running a comparison between a feed forward network without pruning, with importance based pruning and with randomised pruning -

- Run the driver with 3 different configs:
  - Set _total_epochs_ as 100, _inital_iterations_ as 100 and _increment_ as 10 and run the driver. (Without any pruning) and log file enabled in _driver.py_ is _training_logwithoutPruning.txt_
  - Now set _total_epochs_ as 100, _inital_iterations_ as 30 and _increment_ as 10, run the driver and make sure _contribBasedPruning_ set as True (With importance based pruning) and the log file enabled in _driver.py_ is _training_logwithPruning.txt_
  - Now set _total_epochs_ as 100, _inital_iterations_ as 30 and _increment_ as 10, run the driver and make sure _contribBasedPruning_ set as False (With importance based pruning) and log file enabled in _driver.py_ is _training_logworstPruning.txt_
- Run _plot.py_ to generate the plots for viewing the pyplots for accuracy.

## Reports

For the interested reader, an ICLR-style academic report is available under the _reports_ folder. 