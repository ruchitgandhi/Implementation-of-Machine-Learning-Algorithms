import numpy as np

my_data = np.genfromtxt('pca-data.txt', delimiter='')
total=np.sum(my_data,axis=0)
mean= total/my_data.shape[0]

for i in range( my_data.shape[0] ):
    my_data[i,:] = my_data[i,:] - mean

my_data_trans = my_data.transpose()
covariance_matrix = np.cov(my_data_trans)

eigvals, eigvecs = np.linalg.eig(covariance_matrix)

eig_pairs = [(np.abs(eigvals[i]), eigvecs[:,i]) for i in range(len(eigvals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)


truncatedMatrix= np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
print("Directions are:")
print(truncatedMatrix)


print("Eigen Values are:")
print(eig_pairs[0][0])
print(eig_pairs[1][0])


my_data_new = np.zeros((2,6000))


for j in range(6000):
    (my_data_new[:, j])=np.dot(truncatedMatrix.transpose(), my_data_trans[:,j])



data_final = my_data_new.transpose()
print("Transformed Data points are")
print(data_final)

