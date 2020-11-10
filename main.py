from arffdatasetreader import dataset_reader as dr
from dimensionalityreductors import pca
from utils import plotter
from clusteringgenerators import kmeans
from validators.metrics import compute_pca_and_tsne_on_reduced_dataset, compute_pca_and_tsne


if __name__ == '__main__':
    datasets_preprocessed = dr.get_datasets()
    targets_labels = dr.get_datasets_target()

    # plotter.plot_original_dataset(datasets_preprocessed[0])
    # plotter.plot_original_dataset(datasets_preprocessed[1])
    # plotter.plot_original_dataset(datasets_preprocessed[2])

    # pca.test_pca(datasets_preprocessed)

    # kmeans.get_best_k_for_all_datasets_reduced(datasets_preprocessed)
    compute_pca_and_tsne_on_reduced_dataset(datasets_preprocessed, targets_labels, plot_pca_2D=True, plot_tsne_2D=True,
                                          plot_pca_3D=True, plot_tsne_3D=True)
    compute_pca_and_tsne(datasets_preprocessed, plot_pca_2D=True, plot_tsne_2D=True, plot_pca_3D=True, plot_tsne_3D=True)



