from arffdatasetreader import dataset_reader as dr
from dimensionalityreductors import pca
from clusteringgenerators import kmeans
from utils import plotter
import time
from dimensionalityreductors import pca_ipca
import math
from validators.metrics import compute_pca_and_tsne_on_reduced_dataset, compute_pca_and_tsne


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


def test_and_plot_different_params_tsne():
    #TODO
    perplexity = [5, 10, 20, 30, 40, 50]
    plotter.plot_tsne_3D(datasets_preprocessed[0], targets_labels[0], plot_title="",
                                 perplexity=30, learning_rate=200, n_iter=300, random_state=0)


if __name__ == '__main__':
    datasets_preprocessed = dr.get_datasets()
    targets_labels = dr.get_datasets_target()

    # plotter.plot_original_dataset(datasets_preprocessed[0])
    # plotter.plot_original_dataset(datasets_preprocessed[1])
    # plotter.plot_original_dataset(datasets_preprocessed[2])

    test_pca(datasets_preprocessed)
    test_pca_sklearn(datasets_preprocessed)
    test_ipca(datasets_preprocessed)

    # kmeans.get_best_k_for_all_datasets_reduced(datasets_preprocessed)

    compute_pca_and_tsne_on_reduced_dataset(datasets_preprocessed, targets_labels, plot_implemented_pca_2D=True,
                                            plot_implemented_pca_3D=True, plot_sklearn_pca_2D=True, 
                                            plot_sklearn_pca_3D=True, plot_tsne_2D=True, plot_tsne_3D=True)
    
    compute_pca_and_tsne(datasets_preprocessed, plot_implemented_pca_2D=True,
                        plot_implemented_pca_3D=True, plot_sklearn_pca_2D=True,
                        plot_sklearn_pca_3D=True, plot_tsne_2D=True, plot_tsne_3D=True)
