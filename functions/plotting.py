import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import scatter

from functions.objects import PersistenceData
from pandas.plotting import scatter_matrix


def plot_cluster_scatters(clustering, pers_data, method, sample_count):
    cluster_ids = np.unique(clustering)
    method_name = method.__str__().split("(")[0]
    n_clusters = len(cluster_ids)

    # Set up figure and gridspec: reserve final column for colorbar
    fig = plt.figure(figsize=(6 * sample_count, 2 * n_clusters))
    gs = gridspec.GridSpec(n_clusters, sample_count + 1, width_ratios=[1] * sample_count + [0.05], wspace=0.15, hspace=0.2)

    axs = np.empty((n_clusters, sample_count), dtype=object)

    # Fill subplots
    for i, cluster_id in enumerate(cluster_ids):
        cluster = np.where(clustering == cluster_id)[0]
        samples = np.random.choice(cluster, size=min(sample_count, len(cluster)), replace=False)

        for j, sample in enumerate(samples):
            ax = fig.add_subplot(gs[i, j])
            axs[i, j] = ax
            scatter = pers_data.plot_raw_path(ax, sample, colour_bar=False)
            ax.set_title("")

            if j > 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel(f"Cluster {i}\nLatitude")

            if i < n_clusters - 1:
                ax.set_xlabel("")

    # Add colorbar in final column
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Time Step")

    fig.text(0.5, 0.99, f"Paths of {method_name} clusters", ha='center', va='top', fontsize=20)

    title_space_inches = 0.4 + 0.02*n_clusters # this works for some reason
    fig_height_inches = fig.get_figheight()
    top_margin = 1 - (title_space_inches / fig_height_inches)  # relative coordinate

    fig.subplots_adjust(top=top_margin)

def plot_cluster_pers_diagrams(clustering, pers_data: PersistenceData, method, sample_count):
    cluster_ids = np.unique(clustering)
    method_name = method.__str__().split("(")[0]
    n_clusters = len(cluster_ids)

    # Set up figure and gridspec: reserve final column for colorbar
    fig = plt.figure(figsize=(6 * sample_count, 2 * n_clusters))
    gs = gridspec.GridSpec(n_clusters, sample_count + 1, width_ratios=[1] * sample_count + [0.05], wspace=0.2, hspace=0.2)

    axs = np.empty((n_clusters, sample_count), dtype=object)

    # Fill subplots
    for i, cluster_id in enumerate(cluster_ids):
        cluster = np.where(clustering == cluster_id)[0]
        samples = np.random.choice(cluster, size=min(sample_count, len(cluster)), replace=False)

        for j, sample in enumerate(samples):
            ax = fig.add_subplot(gs[i, j])
            axs[i, j] = ax
            pers_data.plot_persistence(ax, sample)
            ax.set_title("")
            ax.locator_params(axis='x', nbins=6)

            if j > 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel(f"Cluster {i}\nDeath")

            if i < n_clusters - 1:
                ax.set_xlabel("")

    fig.text(0.5, 0.99, f"Paths of {method_name} clusters", ha='center', va='top', fontsize=20)

    title_space_inches = 0.4 + 0.02*n_clusters # this works for some reason
    fig_height_inches = fig.get_figheight()
    top_margin = 1 - (title_space_inches / fig_height_inches)  # relative coordinate

    fig.subplots_adjust(top=top_margin)


def plot_scatter_matrix(data):
    size = data.shape[1]
    column_names = data.columns

    scatter_matrix(data, figsize=(size, size))
    plt.suptitle("Scatter Plot Matrix")
    plt.show()