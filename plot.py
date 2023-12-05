import matplotlib.pyplot as plt

def parse_accuracy_data(file_path):
    accuracies = []
    epoch_accuracies = []
    current_epoch = -1

    with open(file_path, 'r') as file:
        for line in file:
            if "Epoch:" in line:
                # Extract epoch number and accuracy from the line
                parts = line.split(',')
                epoch = int(parts[0].split(' ')[-1].strip())
                accuracy_str = parts[-1].strip().split(' ')[-1].replace('%', '')
                accuracy = float(accuracy_str) / 100.0

                # Check if this is a new epoch
                if epoch != current_epoch:
                    if current_epoch != -1:
                        # Calculate and store the average accuracy for the completed epoch
                        epoch_avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
                        accuracies.append(epoch_avg_accuracy)
                        epoch_accuracies = []

                    current_epoch = epoch

                epoch_accuracies.append(accuracy)

        # Don't forget to add the average of the last epoch
        if epoch_accuracies:
            epoch_avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
            accuracies.append(epoch_avg_accuracy)

    return accuracies

def plot_accuracies(*accuracy_lists):
    for accuracies in accuracy_lists:
        plt.plot(accuracies)
    plt.title('Epoch-wise Average Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.legend(['withoutPruning', 'withPruning', 'worstPruning'])
    plt.show()

# Paths to the log files
file_paths = ['training_logwithoutPruning.txt', 'training_logwithPruning.txt', 'training_logworstPruning.txt']

# Parse the data and plot it
accuracy_data_sets = [parse_accuracy_data(path) for path in file_paths]
plot_accuracies(*accuracy_data_sets)
