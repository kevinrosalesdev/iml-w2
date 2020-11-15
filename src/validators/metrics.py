import math
import time

from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

from clusteringgenerators import kmeans
from utils import plotter
from dimensionalityreductors import pca


def compute_pca_and_tsne_on_reduced_dataset(datasets, target_labels=None, plot_implemented_pca_2D=False,
                                            plot_implemented_pca_3D=False, plot_sklearn_pca_2D=False,
                                            plot_sklearn_pca_3D=False, plot_tsne_2D=False, plot_tsne_3D=False,
                                            tsne_n_iterations=None, tsne_learning_rate=None, tsne_perplexity=None):
    datasets_reduced = []
    for dataset in datasets:
        data = pca.apply_dimensionality_reduction(dataset,
                                                  num_components=None,
                                                  print_cov_matrix=True,
                                                  print_eigen=True,
                                                  print_variance_explained=True,
                                                  print_selected_eigen=True,
                                                  plot_transformed_data=False,
                                                  plot_original_data=False
                                                  )
        datasets_reduced.append(data[0])
    best_k_reduced = [9, 20, 11]
    plot_pca_and_tsne(datasets_reduced, best_k_reduced, target_labels, plot_implemented_pca_2D,
                      plot_implemented_pca_3D=plot_implemented_pca_3D, plot_sklean_pca_2D=plot_sklearn_pca_2D,
                      plot_sklearn_pca_3D=plot_sklearn_pca_3D, plot_tsne_2D=plot_tsne_2D, plot_tsne_3D=plot_tsne_3D,
                      tsne_n_iterations=tsne_n_iterations, tsne_learning_rate=tsne_learning_rate,
                      tsne_perplexity=tsne_perplexity)


def compute_pca_and_tsne(datasets, plot_implemented_pca_2D=False, plot_implemented_pca_3D=False,
                         plot_sklearn_pca_2D=False, plot_sklearn_pca_3D=False, plot_tsne_2D=False, plot_tsne_3D=False,
                         tsne_n_iterations=None, tsne_learning_rate=None, tsne_perplexity=None):
    best_k_kmeans_first_assignment = [9, 19, 6]
    plot_pca_and_tsne(datasets, best_k_kmeans_first_assignment, None, plot_implemented_pca_2D,
                      plot_implemented_pca_3D=plot_implemented_pca_3D, plot_sklean_pca_2D=plot_sklearn_pca_2D,
                      plot_sklearn_pca_3D=plot_sklearn_pca_3D, plot_tsne_2D=plot_tsne_2D, plot_tsne_3D=plot_tsne_3D,
                      tsne_n_iterations=tsne_n_iterations, tsne_learning_rate=tsne_learning_rate,
                      tsne_perplexity=tsne_perplexity)


def plot_pca_and_tsne(datasets, best_k, target_labels=None, plot_implemented_pca_2D=False,
                      plot_implemented_pca_3D=False, plot_sklean_pca_2D=False, plot_sklearn_pca_3D=False,
                      plot_tsne_2D=False, plot_tsne_3D=False, tsne_n_iterations=None, tsne_learning_rate=None,
                      tsne_perplexity=None):
    dataset_names = ["Pen-based (num)", "Kropt (cat)", "Hypothyroid (mxd)"]
    for index in range(0, len(datasets)):
        tic = time.time()
        pred_labels, iteration_distance, _ = kmeans.apply_unsupervised_learning(datasets[index],
                                                                                best_k[index],
                                                                                max_iterations=30,
                                                                                use_default_seed=True,
                                                                                plot_distances=False)
        toc = time.time()
        print("--------", dataset_names[index], "--------")
        print(f"K-MEANS execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
        print("Silhouette score", silhouette_score(datasets[index], pred_labels))
        print("Calinski-Harabasz score", calinski_harabasz_score(datasets[index], pred_labels))
        print("Davies Bouldin score", davies_bouldin_score(datasets[index], pred_labels))

        if target_labels != None:
            plotter.plot_confusion_matrix(target_labels[index], pred_labels,
                                      plot_title=f"{dataset_names[index]} - K={best_k[index]}",
                                      is_real_k=False)
        if plot_implemented_pca_2D:
            plotter.plot_implemented_pca_2D(datasets[index], pred_labels,
                                    plot_title=f"Implemented PCA - {dataset_names[index]} - K={best_k[index]}")
        if plot_implemented_pca_3D:
            plotter.plot_implemented_pca_3D(datasets[index], pred_labels,
                                        plot_title=f"Implemented PCA - {dataset_names[index]} - K={best_k[index]}")

        if plot_sklean_pca_2D:
            plotter.plot_sklearn_pca_2D(datasets[index], pred_labels,
                                        plot_title=f"SKLearn PCA - {dataset_names[index]} - K={best_k[index]}")

        if plot_sklearn_pca_3D:
            plotter.plot_sklearn_pca_3D(datasets[index], pred_labels,
                                        plot_title=f"SKLearn PCA - {dataset_names[index]} - K={best_k[index]}")

        if plot_tsne_2D:
            if tsne_n_iterations is None:
                n_iterations = 1000
            else:
                n_iterations = tsne_n_iterations[index]
            if tsne_learning_rate is None:
                tsne_learning_rate = 200
            else:
                lerning_rate = tsne_learning_rate[index]
            if tsne_perplexity is None:
                tsne_perplexity = 30
            else:
                perplexity = tsne_perplexity[index]

            tic = time.time()
            plotter.plot_tsne_2D(datasets[index], pred_labels,
                                 plot_title=f"t-SNE - {dataset_names[index]} - K={best_k[index]}", perplexity=perplexity,
                                 learning_rate=lerning_rate, n_iter=n_iterations, random_state=0)
            toc = time.time()
            print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

        if plot_tsne_3D:
            tic = time.time()
            plotter.plot_tsne_3D(datasets[index], pred_labels,
                                 plot_title=f"t-SNE - {dataset_names[index]} - K={best_k[index]}", perplexity=perplexity,
                                 learning_rate=lerning_rate, n_iter=n_iterations, random_state=0)
            toc = time.time()
            print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")