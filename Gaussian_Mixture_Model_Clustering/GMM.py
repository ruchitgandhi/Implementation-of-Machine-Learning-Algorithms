import numpy as np
import random
from math import pi, exp

num_clusters = 3


def file_read():
    file_data = np.genfromtxt('clusters.txt', delimiter=',')
    return file_data


def normalize_membership(value1, value2, value3):
    total = value1 + value2 + value3
    return value1 / total, value2 / total, value3 / total


# Initialization step to calculate membership from random values
def calculate_membership_matrix(num_points):
    membership_matrix = np.zeros((num_points, 3))

    for i in range(num_points):
        membership_1 = random.random()
        membership_2 = random.random()
        membership_3 = random.random()
        membership_1, membership_2, membership_3 = normalize_membership(membership_1, membership_2, membership_3)
        membership_matrix[i][0] = membership_1
        membership_matrix[i][1] = membership_2
        membership_matrix[i][2] = membership_3

    return membership_matrix


def sum_of_frequency(frequency_array):
    sum = 0
    for i in range(len(frequency_array)):
        sum += frequency_array[i]

    return sum


# calculate amplitude from membership matrix
def calc_amplitude(membership_matrix):
    amplitude_list = [0, 0, 0]
    numerator = 0
    for x in range(num_clusters):
        for y in range(len(membership_matrix[x])):
            numerator += membership_matrix[x][y]
        amplitude_list[x] = numerator / len(membership_matrix[x])
        numerator = 0
    return amplitude_list


# calculate normal distribution given the data, means and covariances
def calculate_normal_distribution(data_vector, mean_for_cluster, cov_for_cluster):
    determinant = np.linalg.det(cov_for_cluster)
    d = len(cov_for_cluster)
    denominator_normal = ((2.0 * pi) ** (d / 2.0)) * (determinant ** (0.5))
    temp = data_vector - mean_for_cluster
    numerator_normal = exp(-0.5 * np.dot(np.dot(temp.transpose(), np.linalg.inv(cov_for_cluster)), temp))
    normal_distribution = numerator_normal / denominator_normal
    return normal_distribution


# Maximization step which calculates membership matrix using normal distribution
def M_step(amplitude_array, mean_matrix, covariance_matrix, input_data, number_of_points):
    new_membership_matrix = np.zeros((number_of_points, num_clusters))
    for i in range(number_of_points):
        denominator_membership = 0
        for j in range(num_clusters):
            new_membership_matrix[i][j] = amplitude_array[j] * calculate_normal_distribution(input_data[i],
                                                                                             mean_matrix[j],
                                                                                             covariance_matrix[j])
            denominator_membership += new_membership_matrix[i][j]
        for j in range(num_clusters):
            new_membership_matrix[i][j] = new_membership_matrix[i][j] / denominator_membership

    return new_membership_matrix.transpose()


# Expectation step which calculates the amplitude, mean and covariance
def E_step(membership_matrix, input_data):
    mean = []
    covariance = []
    for i in range(num_clusters):

        # mean
        frequency = membership_matrix[i]
        numerator_mean = np.dot(frequency, input_data)
        denominator = sum_of_frequency(frequency)
        mean.append(numerator_mean / denominator)

        # covariance
        numerator_cov = 0
        for j in range(number_of_points):
            numerator_cov += frequency[j] * np.outer(input_data[j] - mean[i], input_data[j] - mean[i])

        covariance.append(numerator_cov / denominator)

    return calc_amplitude(membership_matrix), mean, covariance


data = file_read()
number_of_points = data.shape[0]
ric_matrix = calculate_membership_matrix(number_of_points).transpose()
for i in range(100):
    amplitude, mean, covariance = E_step(ric_matrix, data)
    ric_matrix = M_step(amplitude, mean, covariance, data, number_of_points)

print("AMPLITUDE : ", "\n")
print(amplitude)
print("MEAN : ", "\n")
for i in range(len(mean)):
    print(mean[i], "\n")
print("COVARIANCE : ", "\n")
for i in range(len(covariance)):
    print(covariance[i], "\n")
