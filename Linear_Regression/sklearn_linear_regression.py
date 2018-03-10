import numpy as np
from sklearn import linear_model


def file_read():
    return np.genfromtxt('linear-regression.txt', delimiter=',')


data = file_read()
data = np.append(np.ones((3000, 1)), data, axis=1)
y_vector = data[:, 3]
data_matrix = data[:, :3]
regr = linear_model.LinearRegression()
regr.fit(data_matrix, y_vector)
print(regr.coef_)
