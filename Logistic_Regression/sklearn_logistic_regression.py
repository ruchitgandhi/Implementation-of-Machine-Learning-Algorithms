import numpy as np
from sklearn.linear_model import LogisticRegression
def file_read():
    return np.genfromtxt('classification.txt', delimiter=',')

# dimensions 2000 X 5
data = file_read()

# add an additional feature with 1's to each data point to adjust for the constant term
# in the equation of line for linear regression
# dimensions 2000 X 6
data = np.append(np.ones((data.shape[0], 1)), data, axis=1)

# dimensions 2000 X 1
initial_y_vector = data[:, 5]

# dimensions 4 X 2000
data_matrix = data[:, :4]

regr = LogisticRegression()
regr.fit(data_matrix, initial_y_vector)
print(regr.coef_)
y_vector = regr.predict(data_matrix)

# If value of prediction is greater than 0.5 than assign it to +1 class, otherwise -1 class
for i in range(y_vector.shape[0]):
    if y_vector[i] > 0.5:
        y_vector[i] = 1.0
    else:
        y_vector[i] = -1.0

difference_count = 0
for i in range(y_vector.shape[0]):
    if y_vector[i] != initial_y_vector[i]:
        difference_count += 1

# Print number of misclassifications - Comparing the final prediction values to given class values
print(difference_count)
