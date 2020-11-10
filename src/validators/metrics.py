import math
import time

from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

from clusteringgenerators import kmeans
from utils import plotter
from dimensionalityreductors import pca


def compute_pca_and_tsne_on_reduced_dataset(datasets, target_labels=None, plot_pca_2D=False, plot_tsne_2D=False,
                                            plot_pca_3D=False, plot_tsne_3D=False):
    datasets_reduced = []
    for dataset in datasets:
        data = pca.apply_dimensionality_reduction(dataset,
                                                  num_components=None,
                                                  print_cov_matrix=True,
                                                  print_eigen=True,
                                                  print_variance_explained=True,
                                                  print_selected_eigen=True,
                                                  plot_transformed_data=False,
                                                  plot_original_data=False,
                                                  )
        datasets_reduced.append(data[0])
    best_k_reduced = [9, 3, 11]
    plot_pca_and_tsne(datasets_reduced, best_k_reduced, target_labels, plot_pca_2D, plot_tsne_2D, plot_pca_3D, plot_tsne_3D)


def compute_pca_and_tsne(datasets, plot_pca_2D=False, plot_tsne_2D=False, plot_pca_3D=False, plot_tsne_3D=False):
    best_k_kmeans_first_assignment = [9, 19, 6]
    plot_pca_and_tsne(datasets, best_k_kmeans_first_assignment, None, plot_pca_2D, plot_tsne_2D, plot_pca_3D, plot_tsne_3D)


def plot_pca_and_tsne(datasets, best_k, target_labels=None, plot_pca_2D=False, plot_tsne_2D=False,
                      plot_pca_3D=False, plot_tsne_3D=False):
    dataset_names = ["Pen-based (num)", "Kropt (cat)", "Hypothyroid (mxd)"]
    for index in range(0, len(best_k)):
        tic = time.time()
        pred_labels, iteration_distance, _ = kmeans.apply_unsupervised_learning(datasets[index],
                                                                                best_k[index],
                                                                                max_iterations=30,
                                                                                use_default_seed=True,
                                                                                plot_distances=False)
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
        print("Silhouette score", silhouette_score(datasets[index], pred_labels))
        print("Calinski-Harabasz score", calinski_harabasz_score(datasets[index], pred_labels))
        print("Davies Bouldin score", davies_bouldin_score(datasets[index], pred_labels))

        if target_labels != None:
            plotter.plot_confusion_matrix(target_labels[index], pred_labels,
                                      plot_title=f"{dataset_names[index]} - K={best_k[index]}",
                                      is_real_k=False)
        if plot_pca_2D:
            plotter.plot_pca_2D(datasets[index], pred_labels,
                            plot_title=f"{dataset_names[index]} - K={best_k[index]}")
        if plot_tsne_2D:
            tic = time.time()
            plotter.plot_tsne_2D(datasets[index], pred_labels,
                                 plot_title=f"{dataset_names[index]} - K={best_k[index]}")
            toc = time.time()
            print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")