import numpy as np
import csv

def calculate_mean_sd(data):
    iris_classes = np.unique(data[:, -1])  # Assuming the last column contains the class labels
    result = {}

    for iris_class in iris_classes:
        class_data = data[data[:, -1] == iris_class, :-1].astype(float)
        class_mean = np.mean(class_data, axis=0)
        class_sd = np.std(class_data, axis=0)
        class_min = np.min(class_data, axis=0)
        class_max = np.max(class_data, axis=0)

        result[iris_class] = {'mean': class_mean, 'sd': class_sd}

        # Print mean and standard deviation for each class
        print(f"Iris Class: {iris_class}")
        print(f"Mean: {class_mean}")
        print(f"Standard Deviation: {class_sd}")
        print(f"Minimum: {class_min}")
        print(f"Maximum: {class_max}")
        print()

    return result

def preprocess_data(data, mean_sd_dict, num_sd=1):
    filtered_data = []

    for row in data:
        iris_class = row[-1]
        features = row[:-1].astype(float)

        class_mean = mean_sd_dict[iris_class]['mean']
        class_sd = mean_sd_dict[iris_class]['sd']

        # Preprocess the data based on mean ± num_sd * sd
        condition = np.logical_and(features >= class_mean - num_sd * class_sd,
                                   features <= class_mean + num_sd * class_sd)

        # Replace values based on the conditions
        result = np.where(condition, 'M', 'S')
        result = np.where(features > class_mean + num_sd * class_sd, 'L', result)

        # Append the row to filtered_data
        filtered_data.append(np.append(result, iris_class))

    return np.array(filtered_data)

# Read data from iris.data file
with open('iris.data', 'r') as file:
    data_str = file.read()

# Parse the data into a NumPy array
data = np.array([line.split(',') for line in data_str.strip().split('\n')])

# Add header to the data
header = "sepal_length sepal_width petal_length petal_width class"
data_with_header = np.vstack([header.split(' '), data])

# Calculate mean and standard deviation for each class
mean_sd_dict = calculate_mean_sd(data_with_header[1:])

# Preprocess data by filtering out points outside mean ± 2 * sd range for each class
preprocessed_data = preprocess_data(data_with_header[1:], mean_sd_dict, num_sd=2)

# Define the output CSV file name
output_csv_file = 'preprocessed_iris.csv'

# Write preprocessed data to CSV file
with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Write header
    csvwriter.writerow(header.split())

    # Write preprocessed data
    for row in preprocessed_data:
        csvwriter.writerow(row)

print(f"Preprocessed data has been written to '{output_csv_file}'.")

# Write preprocessed data to TXT file with fixed-width spacing and header
output_txt_file = 'preprocessed_iris.txt'
np.savetxt(output_txt_file, preprocessed_data, fmt='%s %s %s %s %s', delimiter=' ', header=header)

print(f"Preprocessed data has been written to '{output_txt_file}'.")
