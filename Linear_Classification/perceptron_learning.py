import numpy as np

my_data = np.genfromtxt('classification.txt', delimiter=',')
my_data = np.delete(my_data, 4, 1)

my_data = np.insert(my_data, 0, 1, 1)
data_points = np.delete(my_data, 4, 1)

weights = np.array([0.1, 0.1, 0.1, 0.1])
alpha = 0.1
count = 0

for z in range(1500):
    count = 0
    for x in range(data_points.shape[0]):
        predict = 0
        y = np.dot(weights, data_points[x].transpose())
        if (y > 0):
            predict = 1
        else:
            predict = -1

        if (predict != my_data[x][4]):
            count += 1
            if (y < 0):
                weights = weights + alpha * data_points[x]
            elif (y > 0):
                weights = weights - alpha * data_points[x]

print("Weights after the final iteration are ", weights)
print("Accuracy after the final iteration is ", (2000 - count) / 20)
