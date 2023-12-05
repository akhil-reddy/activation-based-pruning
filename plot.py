import matplotlib.pyplot as plt

def parse_accuracy_data(file_path):
    accuracies = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Accuracy" in line:
                # Extract accuracy from the line
                parts = line.split(',')
                accuracy_str = parts[-1].strip().split(' ')[-1].replace('%', '')
                accuracies.append(float(accuracy_str) / 100.0) 
    return accuracies

def plot_accuracies(*accuracy_lists):
    for accuracies in accuracy_lists:
        plt.plot(accuracies)
    plt.title('Epoch-wise Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['withoutPruning', 'withPruning', 'worstPruning' ])
    plt.show()

# Paths to the log files
file_paths = [ 'training_logwithoutPruning.txt', 'training_logwithPruning.txt', 'training_logworstPruning.txt']

# Parse the data and plot it
accuracy_data_sets = [parse_accuracy_data(path) for path in file_paths]
plot_accuracies(*accuracy_data_sets)