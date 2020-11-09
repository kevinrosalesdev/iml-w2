from arffdatasetreader import dataset_reader as dr
from clusteringgenerators import kmeans
from dimensionalityreductors import pca
from utils import plotter
import time
import math


def plot_original_dataset(dataset):
    np_dataset = dataset.to_numpy()
    plotter.plot_two_features(np_dataset[:, 0], np_dataset[:, 1])
    plotter.plot_three_features(np_dataset[:, 0], np_dataset[:, 1], np_dataset[:, 2])


def test_pca(datasets):
    print("Applying PCA to Numerical Dataset: Pen-based...")
    tic = time.time()
    pca.apply_dimensionality_reduction(datasets[0],
                                       num_components=2,
                                       print_cov_matrix=True,
                                       print_eigen=True,
                                       print_selected_eigen=True,
                                       plot_transformed_data=True,
                                       plot_original_data=True)
    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")

    print("Applying PCA to Numerical Dataset: Kropt...")
    tic = time.time()
    pca.apply_dimensionality_reduction(datasets[1],
                                       num_components=2,
                                       print_cov_matrix=True,
                                       print_eigen=True,
                                       print_selected_eigen=True,
                                       plot_transformed_data=True,
                                       plot_original_data=True)
    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")

    print("Applying PCA to Numerical Dataset: Hypothyroid...")
    tic = time.time()
    pca.apply_dimensionality_reduction(datasets[2],
                                       num_components=2,
                                       print_cov_matrix=True,
                                       print_eigen=True,
                                       print_selected_eigen=True,
                                       plot_transformed_data=True,
                                       plot_original_data=True)
    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")

def test_pca_sklearn(datasets):
    print("Applying PCA with Sklearn to Numerical Dataset: Pen-based...")
    tic = time.time()
    pca_ipca.pca_sklearn(datasets[0], n_components=2)

    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")

    print("Applying PCA with Sklearn to Numerical Dataset: Kropt...")
    tic = time.time()
    pca_ipca.pca_sklearn(datasets[1], n_components=2)

    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")

    print("Applying PCA with Sklearn to Numerical Dataset: Hypothyroid...")
    tic = time.time()
    pca_ipca.pca_sklearn(datasets[2], n_components=2)

def test_ipca(datasets):
    print("Applying IPCA to Numerical Dataset: Pen-based...")
    tic = time.time()
    pca_ipca.ipca(datasets[0], n_components=2)

    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")

    print("Applying IPCA to Numerical Dataset: Kropt...")
    tic = time.time()
    pca_ipca.ipca(datasets[1], n_components=2)

    toc = time.time()
    print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s "
          f"{(math.trunc((toc - tic) * 1000) % 1000)}ms")

    print("Applying IPCA to Numerical Dataset: Hypothyroid...")
    tic = time.time()
    pca_ipca.ipca(datasets[2], n_components=2)


if __name__ == '__main__':
    datasets_preprocessed = dr.get_datasets()
    targets_labels = dr.get_datasets_target()

    plot_original_dataset(datasets_preprocessed[0])
    plot_original_dataset(datasets_preprocessed[1])
    plot_original_dataset(datasets_preprocessed[2])

    test_pca(datasets_preprocessed)
    test_pca_sklearn(datasets_preprocessed)
    test_ipca(datasets_preprocessed)
