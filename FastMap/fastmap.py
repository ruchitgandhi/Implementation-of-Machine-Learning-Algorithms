import numpy as np
from random import randint
from math import sqrt
import matplotlib.pyplot as plt

iterations_for_max_distance_heuristic = 5
number_of_points = 10
dimensions = 2

# initialize coordinates matrix
coordinates_2D = np.zeros((number_of_points, dimensions))


def file_read():
    return np.genfromtxt('fastmap-data.txt', delimiter='\t')

# read given wordlist
def read_wordlist():
    with open('fastmap-wordlist.txt') as f:
        content = f.readlines()
    words = [x.strip() for x in content]
    return words

# form distance matrix from given input data
def form_distance_matrix(input_data):
    result = np.zeros((10, 10))
    for i in range(len(input_data)):
        first_point = int(input_data[i][0] - 1)
        second_point = int(input_data[i][1] - 1)
        result[first_point][second_point] = input_data[i][2]
        result[second_point][first_point] = input_data[i][2]

    return result

# given a pivot find the farthest point from it
def find_farthest_point(distances):
    max_distance = 0.0
    index_of_max = 0
    for i in range(len(distances)):
        if max_distance < distances[i]:
            max_distance = distances[i]
            index_of_max = i

    return max_distance, index_of_max


# find farthest pair in distance matrix using heuristic
def find_farthest_pair(distance_matrix):
    max_distance = 0.0
    farthest_pair = (0, 0)
    pivot = randint(0, 9)
    for i in range(iterations_for_max_distance_heuristic):
        distance, point = find_farthest_point(distance_matrix[pivot])
        if max_distance < distance:
            farthest_pair = pivot, point
            max_distance = distance
        elif max_distance == distance:
            # select the one with the minimum object ID
            if (point < farthest_pair[0] and point < farthest_pair[1]) or (
                    pivot < farthest_pair[0] and pivot < farthest_pair[1]):
                farthest_pair = pivot, point

        pivot = point

    return farthest_pair

# update distance matrix us
def update_distance_matrix(latest_coordinate):
    for i in range(len(distance_matrix)):
        j = i + 1
        while j < number_of_points:
            distance_matrix[i][j] = sqrt(distance_matrix[i][j] ** 2 - \
                                         (coordinates_2D[i][latest_coordinate] - coordinates_2D[j][
                                             latest_coordinate]) ** 2)
            distance_matrix[j][i] = distance_matrix[i][j]
            j += 1


data = file_read()
distance_matrix = form_distance_matrix(data)

for k in range(dimensions):
    point1, point2 = find_farthest_pair(distance_matrix)
    # point with smaller object ID is taken as first reference point (a) and second one as b.
    reference_a, reference_b = (point1, point2) if point1 < point2 else (point2, point1)
    dab = distance_matrix[reference_a][reference_b]

    for i in range(number_of_points):
        coordinates_2D[i][k] = \
            (distance_matrix[reference_a][i] ** 2 + dab ** 2 - distance_matrix[i][reference_b] ** 2) / (2 * dab)

    update_distance_matrix(k)

print("COORDINATES in 2D : \n")
print(coordinates_2D)

# plot words in 2D space
wordlist = read_wordlist()
X = coordinates_2D[:, 0]
Y = coordinates_2D[:, 1]
plt.scatter(X, Y)
for i, word in enumerate(wordlist):
    plt.annotate(word, (X[i], Y[i]))
plt.show()
