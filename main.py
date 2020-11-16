from arffdatasetreader import dataset_reader as dr
from dimensionalityreductors import pca
from clusteringgenerators import kmeans
from utils import plotter
from dimensionalityreductors import sklearn_pca_ipca
from validators.metrics import compute_pca_and_tsne_on_reduced_dataset, compute_pca_and_tsne


def test_and_plot_different_params_tsne(dataset, best_k):
    # test on different parameters of t-SNE
    perplexity = [10, 30, 50]
    learning_rate = [10, 200, 1000]
    n_iter = [1000, 3000, 5000]

    # reducing dataset dimensionality
    data = pca.apply_dimensionality_reduction(dataset,
                                              num_components=None,
                                              print_cov_matrix=True,
                                              print_eigen=True,
                                              print_variance_explained=True,
                                              print_selected_eigen=True,
                                              plot_transformed_data=False,
                                              plot_original_data=False,
                                              )
    # getting predicted labels
    pred_labels, _, _ = kmeans.apply_unsupervised_learning(data[0], best_k, max_iterations=30,
                                                           use_default_seed=True, plot_distances=False)

    for n_it in n_iter:
        for lr in learning_rate:
            for per in perplexity:
                title = f"N° iter. = {n_it}, Learn. Rate = {lr}, Perp. = {per}"
                plotter.plot_tsne_2D(data[0], pred_labels, plot_title=title, perplexity=per,
                                     learning_rate=lr, n_iter=n_it, random_state=0)


if __name__ == '__main__':
    # TODO dr.get_datasets() will return a list with the three datasets already preprocessed in order:
    # Numeric (Pen-Based), Categorical (Kropt), Mixed(Hypothyroid).
    datasets_preprocessed = dr.get_datasets()

    # TODO if you find: "index_of_the_dataset" that means that you have to choose the index of the dataset
    # TODO to plot the original dataset selecting the first 2/3 features. !uncomment the code below!
    # plotter.plot_original_dataset(datasets_preprocessed[index_of_the_dataset])

    # TODO to do the pairplot of the original dataset. !uncomment the code below!
    # N.B. with the last dataset the plot crash! (too many features)
    # plotter.sn_plot_original_dataset(datasets_preprocessed[index_of_the_dataset])

    # TODO to test our implementation of the PCA, plot two/three features of sklearn.PCA or sklearn.IPCA. !uncomment the code below!
    # pca.test_pca(datasets_preprocessed)
    # sklearn_pca_ipca.test_pca_sklearn(datasets_preprocessed)
    # sklearn_pca_ipca.test_ipca(datasets_preprocessed)

    # TODO to get the plot of the best k for all datasets with previous reduction or not. !uncomment the code below!
    # kmeans.get_best_k(datasets_preprocessed)
    # kmeans.get_best_k_for_all_datasets_reduced(datasets_preprocessed)

    # These are the chosen values for the t-SNE function
    perplexity = [30, 50, 50]
    learning_rate = [200, 200, 200]
    n_iter = [3000, 3000, 3000]

    # TODO to plot PCA and t-SNE after clustering with K-Means (using the chosen optimal values) for all datasets after PCA reduction
    """
    compute_pca_and_tsne_on_reduced_dataset(datasets_preprocessed, None, plot_implemented_pca_2D=True,
                                            plot_implemented_pca_3D=True, plot_sklearn_pca_2D=True, 
                                            plot_sklearn_pca_3D=True, plot_tsne_2D=True, plot_tsne_3D=True,
                                            tsne_n_iterations=n_iter, tsne_perplexity=perplexity, 
                                            tsne_learning_rate=learning_rate)

    """
    
    # TODO to plot PCA and t-SNE after clustering with K-Means (using the chosen optimal values) for all datasets without PCA reduction
    """
    compute_pca_and_tsne(datasets_preprocessed, plot_implemented_pca_2D=True,
                        plot_implemented_pca_3D=True, plot_sklearn_pca_2D=True,
                        plot_sklearn_pca_3D=True, plot_tsne_2D=True, plot_tsne_3D=True,
                        tsne_n_iterations=n_iter, tsne_perplexity=perplexity, 
                        tsne_learning_rate=learning_rate)
    """

    best_k_reduced = [9, 20, 11]
    # TODO for running the test on the different values for the learning_rate, perplexity, n_iter of the t-SNE. !uncomment the code below!
    # N.B. due to running time issues it's better to run one of these at the time
    # test_and_plot_different_params_tsne(datasets_preprocessed[0], best_k_reduced[0])
    # test_and_plot_different_params_tsne(datasets_preprocessed[1], best_k_reduced[1])
    # test_and_plot_different_params_tsne(datasets_preprocessed[2], best_k_reduced[2])
