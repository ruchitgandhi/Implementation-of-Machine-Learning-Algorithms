import numpy as np
from math import exp

learning_rate = 5.0 * exp(-5)


# Read data from input file
def file_read():
    return np.genfromtxt('classification.txt', delimiter=',')


# Initialize W vectors
def initialize_w(dimensions):
    return np.zeros((dimensions, 1))
    # return np.matrix([[0.1],[0.1],[0.1],[0.1]])


# Given the values of the variables (w), data points (x) and predictions (y) calculate the gradient
# Sigmoid function taken is - e^s/(1+e^s)
def calculate_gradient(w, x, y):
    result = np.zeros((number_of_dimensions, 1))
    for i in range(number_of_points_in_dataset):
        x_vector = x[:, i:i + 1]
        temp = exp(-1.0 * y[i] * np.dot(w, x_vector))
        temp = (-1.0 * temp) / (1.0 + temp)
        result += temp * y[i] * x_vector

    return result / number_of_points_in_dataset


# Apply sigmoid function to given input - e^s/(1 + e^s)
def sigmoidfunction(input):
    return exp(input) / (1.0 + exp(input))


# Calculate sigmoid output for wTranspose.x
def calculate_sigmoid(w, x):
    y = np.zeros((number_of_points_in_dataset, 1))
    for i in range(number_of_points_in_dataset):
        x_vector = x[:, i:i + 1]
        y[i] = sigmoidfunction(np.dot(w, x_vector))

    return y


# dimensions 2000 X 5
data = file_read()
number_of_points_in_dataset = data.shape[0]
# add an additional feature with 1's to each data point to adjust for the constant term
# in the equation of line for linear regression
# dimensions 2000 X 6
data = np.append(np.ones((data.shape[0], 1)), data, axis=1)

# dimensions 2000 X 1
initial_y_vector = data[:, 5]
y_vector = initial_y_vector

# dimensions 4 X 2000
data_matrix = data[:, :4].transpose()
number_of_dimensions = data_matrix.shape[0]
# dimensions 4 X 1
w_vector = initialize_w(data_matrix.shape[0])

for i in range(7000):
    # dimensions 4 X 1
    gradient = calculate_gradient(w_vector.transpose(), data_matrix, y_vector)

    # dimensions 4 X 1
    w_vector = w_vector - learning_rate * gradient
    # Recalculate prediction based on new w values
    # dimensions 2000 X 1
    y_vector = calculate_sigmoid(w_vector.transpose(), data_matrix)

print("Weights are: ")
print(w_vector.transpose())

# If value of prediction is greater than 0.5 than assign it to +1 class, otherwise -1 class
for i in range(y_vector.shape[0]):
    if y_vector[i] > 0.5:
        y_vector[i] = 1.0
    else:
        y_vector[i] = -1.0

accuracy = 0
for i in range(y_vector.shape[0]):
    if y_vector[i] == initial_y_vector[i]:
        accuracy += 1

# Print number of misclassifications - Comparing the final prediction values to given class values
print("Accuracy: ")
print(accuracy*100/number_of_points_in_dataset)

