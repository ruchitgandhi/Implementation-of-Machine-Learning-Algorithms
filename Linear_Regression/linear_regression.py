import numpy as np
from numpy.linalg import inv


# Read data from input file
def file_read():
    return np.genfromtxt('linear-regression.txt', delimiter=',')


# Calculate variables for the linear equation - w
def calculate_W(data_matrix, y_vector):
    temp1 = inv(np.dot(data_matrix, data_matrix.transpose()))
    temp2 = np.dot(data_matrix, y_vector)
    return np.dot(temp1, temp2)


# dimensions 3000 X 3
data = file_read()

# add an additional feature with 1's to each data point to adjust for the constant term
# in the equation of line for linear regression
# dimensions 3000 X 4
data = np.append(np.ones((data.shape[0], 1)), data, axis=1)

# dimensions 3000 X 1
y_vector = data[:, 3]

# dimensions 3 X 3000
data_matrix = data[:, :3].transpose()

# dimensions 3 X 1
result = calculate_W(data_matrix, y_vector)
print(result)
