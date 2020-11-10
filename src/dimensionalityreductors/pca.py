import numpy as np
from utils import plotter


def apply_dimensionality_reduction(dataset, num_components=None,
                                   print_cov_matrix=False, print_eigen=False, print_selected_eigen=False,
                                   print_variance_explained=False,
                                   plot_transformed_data=False, plot_original_data=False):

    np_dataset = dataset.to_numpy()
    d_mean_vector = np.mean(np_dataset, axis=0)
    np_dataset_mean = np.subtract(np_dataset, d_mean_vector)
    cov_matrix = np.cov(np_dataset_mean.T)

    if print_cov_matrix:
        print("Covariance Matrix:\n" + str(cov_matrix))

    # The eigenvalues are not necessarily ordered by default.
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Calculating the explained variance on each of components
    variance_explained = [(eigenvalue / sum(eigenvalues)) * 100 for eigenvalue in eigenvalues]
    cumulative_variance_explained = np.cumsum(variance_explained)

    if print_eigen:
        print("Eigenvalues:\n" + str(eigenvalues))
        if print_variance_explained:
            print("Variance explained:\n" + str(variance_explained))
            print("Cumulative variance explained:\n" + str(cumulative_variance_explained))
        print("Eigenvectors:\n" + str(eigenvectors))

    # If the number of components is not defined, num_components = len(eigenvalues > 1.).
    # If after this computation num_components < 2, then num_components = number of components that explain the 95% of
    # the cumulative variance.
    if num_components is None:
        num_components = np.where(eigenvalues > 1.)[0].shape[0]
        if num_components < 2:
            print("[WARNING] Not enough eigenvalues are > 1., getting num of components where the cumulative "
                  "variance explained is 95%")
            num_components = np.where(cumulative_variance_explained < 95)[0].shape[0]
            print("num_components cumulative=", str(num_components))

    ordered_idx = eigenvalues.argsort()[::-1][:num_components]
    eigenvalues = eigenvalues[ordered_idx]
    eigenvectors = eigenvectors[:, ordered_idx]

    if print_selected_eigen:
        print("Selected eigenvalues (num_components = " + str(num_components) + "):\n" + str(eigenvalues))
        print("Selected eigenvectors (num_components = " + str(num_components) + "):\n" + str(eigenvectors))

    transformed_data = np.matmul(eigenvectors.T, np_dataset_mean.T).T

    if plot_transformed_data:
        plotter.plot_two_features(transformed_data[:, 0], transformed_data[:, 1], title='PCA')

    original_data = np.add(np.matmul(eigenvectors, transformed_data.T).T, d_mean_vector)

    if plot_original_data:
        plotter.plot_two_features(original_data[:, 0], original_data[:, 1])

    return [transformed_data, original_data]
