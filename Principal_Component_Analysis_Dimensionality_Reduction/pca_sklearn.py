from sklearn import decomposition
import numpy as np

data = np.genfromtxt("pca-data.txt", delimiter='')

sklearn_pca = decomposition.PCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(data)
print("Top 2 Eigen Vectors : \n", sklearn_pca.components_)
print("\n Top 2 Eigen Values : \n", sklearn_pca.explained_variance_)