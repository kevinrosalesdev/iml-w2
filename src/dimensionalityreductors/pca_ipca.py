import numpy as np
from utils import plotter
from sklearn.decomposition import PCA, IncrementalPCA
from arffdatasetreader import dataset_reader as dr

"""
class sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto',
tol=0.0, iterated_power='auto', random_state=None)[source]


def apply_dimensionality_reduction(dataset, num_components=None,
                                   print_cov_matrix=False, print_eigen=False, print_selected_eigen=False,
                                   plot_transformed_data=False, plot_original_data=False)
"""
def pca_sklearn(dataset, n_components):

    pca = PCA(n_components).fit(dataset)
    transformed_data = pca.transform(dataset)
    plotter.plot_two_features(transformed_data[:, 0], transformed_data[:, 1], title='PCA SKLEARN')
    return transformed_data

def ipca(dataset, n_components):
    ipca = IncrementalPCA(n_components).fit(dataset)
    transformed_data = ipca.transform(dataset)
    plotter.plot_two_features(transformed_data[:, 0], transformed_data[:, 1], title='IPCA')
    return transformed_data


