from arffdatasetreader import dataset_reader as dr
from dimensionalityreductors import pca
from clusteringgenerators import kmeans
from utils import plotter
from dimensionalityreductors import sklearn_pca_ipca
from validators.metrics import compute_pca_and_tsne_on_reduced_dataset, compute_pca_and_tsne


def test_and_plot_different_params_tsne(dataset, best_k):
    perplexity = [30]
    learning_rate = [10, 1000]
    n_iter = [1000]

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
    pred_labels, __, _ = kmeans.apply_unsupervised_learning(data[0], best_k, max_iterations=30,
                                                            use_default_seed=True, plot_distances=False)

    for n_it in n_iter:
        for lr in learning_rate:
            for per in perplexity:
                title = f"N° iter. = {n_it}, Learn. Rate = {lr}, Perp. = {per}"
                plotter.plot_tsne_2D(data[0], pred_labels, plot_title=title, perplexity=per,
                                     learning_rate=lr, n_iter=n_it, random_state=0)
    """
    DESCRIPTION OF THE PARA
    TNSE (
     n_components: Any = 2,
     perplexity: Any = 30.0,
     # The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. 
     # Larger datasets usually require a larger perplexity. 
     # Consider selecting a value between 5 and 50. Different values can result in significanlty different results.
     
     early_exaggeration: Any = 12.0, 
     # If the cost function increases during initial optimization, 
     # the early exaggeration factor or the learning rate might be too high
     
     learning_rate: Any = 200.0, #in the range [10.0, 1000.0]
     # If the learning rate is too high, the data may look like a ‘ball’ with any point approximately 
     # equidistant from its nearest neighbours. If the learning rate is too low, most points 
     # may look compressed in a dense cloud with few outliers. If the cost function gets stuck in a bad local minimum 
     # increasing the learning rate may help
     
     n_iter: Any = 1000, #Should be at least 250
     n_iter_without_progress: Any = 300, # 50 by 50 
     min_grad_norm: Any = 1e-7,
     metric: Any = "euclidean",
     init: Any = "random",
     verbose: Any = 0,
     random_state: Any = None,
     # Determines the random number generator. Pass an int for reproducible results across multiple function calls. 
     # Note that different initializations might result in different local minima of the cost function
     method: Any = 'barnes_hut', 
     # By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time. 
     # method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time.
     angle: Any = 0.5,
     # This method is not very sensitive to changes in this parameter in the range of 0.2 - 0.8. 
     # Angle less than 0.2 has quickly increasing computation time and angle greater 0.8 has quickly increasing error.
     n_jobs: Any = None) -> Optional[Any]
    """


if __name__ == '__main__':
    # best_k_kmeans_first_assignment = [9, 19, 6]
    datasets_preprocessed = dr.get_datasets()
    targets_labels = dr.get_datasets_target()

    # plotter.plot_original_dataset(datasets_preprocessed[0])
    # plotter.plot_original_dataset(datasets_preprocessed[1])
    # plotter.plot_original_dataset(datasets_preprocessed[2])

    # pca.test_pca(datasets_preprocessed)
    # sklearn_pca_ipca.test_pca_sklearn(datasets_preprocessed)
    # sklearn_pca_ipca.test_ipca(datasets_preprocessed)

    # kmeans.get_best_k_for_all_datasets_reduced(datasets_preprocessed)

    """
    compute_pca_and_tsne_on_reduced_dataset(datasets_preprocessed, targets_labels, plot_implemented_pca_2D=True,
                                            plot_implemented_pca_3D=True, plot_sklearn_pca_2D=True, 
                                            plot_sklearn_pca_3D=True, plot_tsne_2D=True, plot_tsne_3D=True)
    
    compute_pca_and_tsne(datasets_preprocessed, plot_implemented_pca_2D=True,
                        plot_implemented_pca_3D=True, plot_sklearn_pca_2D=True,
                        plot_sklearn_pca_3D=True, plot_tsne_2D=True, plot_tsne_3D=True)
    """
    #TODO Kevin--> run the funcion below for the three dataset --> N.B. save the picture after each run of a dataset
    best_k_reduced = [9, 20, 11]
    test_and_plot_different_params_tsne(datasets_preprocessed[0], best_k_reduced[0])
