from utils import plotter
from sklearn.decomposition import PCA, IncrementalPCA
import time
import math
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
    plotter.plot_two_features(transformed_data[:, 0], transformed_data[:, 1], title='SKLearn PCA')
    return transformed_data

def ipca(dataset, n_components):
    ipca = IncrementalPCA(n_components).fit(dataset)
    transformed_data = ipca.transform(dataset)
    plotter.plot_two_features(transformed_data[:, 0], transformed_data[:, 1], title='IPCA')
    return transformed_data



def test_pca_sklearn(datasets):
    print("Applying PCA with Sklearn to Numerical Dataset: Pen-based...")
    tic = time.time()
    pca_sklearn(datasets[0], n_components=2)

    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")

    print("Applying PCA with Sklearn to Numerical Dataset: Kropt...")
    tic = time.time()
    pca_sklearn(datasets[1], n_components=2)

    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")

    print("Applying PCA with Sklearn to Numerical Dataset: Hypothyroid...")
    tic = time.time()
    pca_sklearn(datasets[2], n_components=2)

    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")


def test_ipca(datasets):
    print("Applying IPCA to Numerical Dataset: Pen-based...")
    tic = time.time()
    ipca(datasets[0], n_components=2)

    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")

    print("Applying IPCA to Numerical Dataset: Kropt...")
    tic = time.time()
    ipca(datasets[1], n_components=2)

    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")

    print("Applying IPCA to Numerical Dataset: Hypothyroid...")
    tic = time.time()
    ipca(datasets[2], n_components=2)

    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")



