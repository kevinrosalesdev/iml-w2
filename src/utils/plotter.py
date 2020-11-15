import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dimensionalityreductors import pca


# External Metric
from sklearn.metrics import confusion_matrix


def plot_error(iteration_distances):
    plt.plot(list(range(0, len(iteration_distances))), iteration_distances)
    plt.ylabel('Sum of distances from each sample to its nearest cluster')
    plt.xlabel('Iteration')
    plt.title('Sum of distances per iteration')
    plt.grid()
    plt.show()


def plot_k_error(k_error):
    plt.plot(list(range(2, len(k_error) + 2)), k_error, 'o-', c='red')
    plt.ylabel('Sum of distances from each sample to its nearest cluster')
    plt.xlabel('K')
    plt.title('Sum of distances for each \'K\' value')
    plt.xticks(list(range(2, len(k_error) + 2)))
    plt.grid()
    plt.show()


def plot_k_silhouette_score(s_scores):
    plt.plot(list(range(2, len(s_scores) + 2)), s_scores, 'o-', c='red')
    plt.ylabel('Silhouette score')
    plt.xlabel('K')
    plt.title('Silhouette score for each \'K\' value')
    plt.xticks(list(range(2, len(s_scores) + 2)))
    plt.grid()
    plt.show()


def plot_k_calinski_harabasz_score(ch_score):
    plt.plot(list(range(2, len(ch_score) + 2)), ch_score, 'o-')
    plt.ylabel('Calinski-Harabasz score')
    plt.xlabel('K')
    plt.title('Calinski-Harabasz score score for each \'K\' value')
    plt.xticks(list(range(2, len(ch_score) + 2)))
    plt.grid()
    plt.show()


def plot_k_davies_bouldin_score(db_score):
    plt.plot(list(range(2, len(db_score) + 2)), db_score, 'o-')
    plt.ylabel('Davies-Bouldin score')
    plt.xlabel('K')
    plt.title('Davies-Bouldin score for each \'K\' value')
    plt.xticks(list(range(2, len(db_score) + 2)))
    plt.grid()
    plt.show()


def plot_confusion_matrix(target, predicted, plot_title='', is_real_k=False):
    conf_matrix = confusion_matrix(target, predicted)
    unique_target = list(set(target))
    unique_predicted = list(set(predicted))
    if len(unique_predicted) < len(unique_target):
        conf_matrix_df = modify_labels_length_drop_zeros(conf_matrix, unique_predicted, unique_target, True)
    elif len(unique_predicted) > len(unique_target):
        conf_matrix_df = modify_labels_length_drop_zeros(conf_matrix, unique_target, unique_predicted, False)
    else:
        conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_target, columns=unique_predicted)

    cmap_value = 'Blues'
    if is_real_k:
        cmap_value = 'Reds'
    sn.heatmap(conf_matrix_df, annot=True, fmt='g', cmap=cmap_value, linewidths=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(plot_title)
    plt.show()


def modify_labels_length_drop_zeros(conf_matrix, list_to_change, list_target, has_less_columns):
    unique_predicted_bigger = list_target.copy()
    for index in range(0, len(list_to_change)):
        unique_predicted_bigger[index] = list_to_change[index]
    conf_matrix_df = pd.DataFrame(conf_matrix, index=list_target, columns=unique_predicted_bigger)
    if has_less_columns:
        conf_matrix_df = conf_matrix_df.loc[:, (conf_matrix_df != 0).any(axis=0)]
    else:
        conf_matrix_df = conf_matrix_df.loc[(conf_matrix_df != 0).any(axis=1)]
    return conf_matrix_df


def plot_sklearn_pca_2D(dataset, labels, plot_title=''):
    pca = PCA(n_components=2)
    df_2D = pd.DataFrame(pca.fit_transform(dataset), columns=['PCA1', 'PCA2'])
    df_2D['Cluster'] = labels
    sn.lmplot(x="PCA1", y="PCA2", data=df_2D, fit_reg=False, hue='Cluster', legend=False, scatter_kws={"s": 1})
    plt.legend(title='Cluster', loc='best', prop={'size': 6})
    plt.title(plot_title)
    plt.show()


def plot_sklearn_pca_3D(dataset, labels, plot_title=''):
    pca = PCA(n_components=3)
    df_3D = pd.DataFrame(pca.fit_transform(dataset), columns=['PCA%i' % i for i in range(3)])
    df_3D['Cluster'] = labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_3D['PCA0'], df_3D['PCA1'], df_3D['PCA2'], c=df_3D.Cluster, s=1)
    plt.legend(*scatter.legend_elements(), title='Cluster', loc='upper left', prop={'size': 6})
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(plot_title)
    plt.show()


def plot_implemented_pca_2D(dataset, labels, plot_title=''):

    data = pca.apply_dimensionality_reduction(dataset,
                                              num_components=2,
                                              print_cov_matrix=True,
                                              print_eigen=True,
                                              print_variance_explained=True,
                                              print_selected_eigen=True,
                                              plot_transformed_data=False,
                                              plot_original_data=False,
                                              )
    df_2D = pd.DataFrame(data[0], columns=['PCA1', 'PCA2'])
    df_2D['Cluster'] = labels
    sn.lmplot(x="PCA1", y="PCA2", data=df_2D, fit_reg=False, hue='Cluster', legend=False, scatter_kws={"s": 1})
    plt.legend(title='Cluster', loc='best', prop={'size': 6})
    plt.title(plot_title)
    plt.show()


def plot_implemented_pca_3D(dataset, labels, plot_title=''):
    data = pca.apply_dimensionality_reduction(dataset,
                                              num_components=3,
                                              print_cov_matrix=True,
                                              print_eigen=True,
                                              print_variance_explained=True,
                                              print_selected_eigen=True,
                                              plot_transformed_data=False,
                                              plot_original_data=False,
                                              )

    df_3D = pd.DataFrame(data[0], columns=['PCA%i' % i for i in range(3)])
    df_3D['Cluster'] = labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_3D['PCA0'], df_3D['PCA1'], df_3D['PCA2'], c=df_3D.Cluster, s=1)
    plt.legend(*scatter.legend_elements(), title='Cluster', loc='upper left', prop={'size': 6})
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(plot_title)
    plt.show()


def plot_tsne_2D(dataset, labels, plot_title='', perplexity=30, learning_rate=200, n_iter=1000, random_state=None):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity,
                learning_rate=learning_rate, n_iter=n_iter,
                random_state=random_state)

    df_2D = pd.DataFrame(tsne.fit_transform(dataset), columns=['T-SNE1', 'T-SNE2'])
    df_2D['Cluster'] = labels
    sn.lmplot(x="T-SNE1", y="T-SNE2", data=df_2D, fit_reg=False, hue='Cluster', legend=False, scatter_kws={"s": 1})
    plt.legend(title='Cluster', loc='best', prop={'size': 6})
    plt.title(plot_title)
    #TODO After usage --> Delete these lines
    plt.tight_layout()
    plt.savefig('pictures/'+plot_title.replace(" ", "") + '.png')
    plt.show()


def plot_tsne_3D(dataset, labels, plot_title='', perplexity=30, learning_rate=200, n_iter=1000, random_state=None):
    tsne = TSNE(n_components=3, verbose=1, perplexity=perplexity,
                learning_rate=learning_rate, n_iter=n_iter,
                random_state=random_state)

    df_3D = pd.DataFrame(tsne.fit_transform(dataset), columns=['TSNE%i' % i for i in range(3)])
    df_3D['Cluster'] = labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_3D['TSNE0'], df_3D['TSNE1'], df_3D['TSNE2'], c=df_3D.Cluster, s=1)
    plt.legend(*scatter.legend_elements(), title='Cluster', loc='upper left', prop={'size': 6})
    ax.set_xlabel("TSNE1")
    ax.set_ylabel("TSNE2")
    ax.set_zlabel("TSNE3")
    ax.set_title(plot_title)
    plt.show()


def plot_two_features(feature_1, feature_2, title='Plot of two features'):
    plt.plot(feature_1, feature_2, '.', markersize=1)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid()
    plt.show()


def plot_three_features(feature_1, feature_2, feature_3, title='Plot of three features'):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(feature_1, feature_2, feature_3, '.', markersize=1)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.title(title)
    plt.grid()
    plt.show()


def plot_original_dataset(dataset):
    np_dataset = dataset.to_numpy()
    plot_two_features(np_dataset[:, 0], np_dataset[:, 1])
    plot_three_features(np_dataset[:, 0], np_dataset[:, 1], np_dataset[:, 2])
