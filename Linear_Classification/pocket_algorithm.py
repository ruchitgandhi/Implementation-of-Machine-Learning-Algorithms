import numpy as np
import matplotlib.pyplot as plt

my_data = np.genfromtxt('classification.txt', delimiter=',')
my_data = np.delete(my_data, 3, 1)

my_data = np.insert(my_data, 0, 1, 1)

data_points = np.delete(my_data, 4, 1)

weights = np.array([0.1, 0.1, 0.1, 0.1])
alpha = 0.1

distance = 2001
minWeights = weights

plotGraph = np.zeros(shape=(7000, 2))

for z in range(7000):

    for x in range(data_points.shape[0]):
        predict = 0
        y = np.dot(weights, data_points[x].transpose())
        if (y > 0):
            predict = 1
        else:
            predict = -1

        if (predict != my_data[x][4]):

            if (y < 0):
                weights = weights + alpha * data_points[x]
            elif (y > 0):
                weights = weights - alpha * data_points[x]

    count = 0
    for a in range(data_points.shape[0]):
        predict1 = 0
        b = np.dot(weights, data_points[a].transpose())
        if (b > 0):
            predict1 = 1
        else:
            predict1 = -1

        if (predict1 != my_data[a][4]):
            count += 1

    plotGraph[z][0] = z
    plotGraph[z][1] = count

    if (count < distance):
        distance = count
        minWeights = weights

print(weights)
print((2000 - count) / 20)
print()
print(minWeights)

print((2000 - distance) / 20)

plt.plot(plotGraph[:, 0], plotGraph[:, 1])
plt.xlabel('No. of Iterations')
plt.ylabel('No. of Mismatches')
plt.xlim([0, 7000])
plt.ylim([0, 2000])
plt.title('Misclassification plot')
plt.show()
