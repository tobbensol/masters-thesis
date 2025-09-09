import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import scatter

from functions.objects import PersistenceData
from pandas.plotting import scatter_matrix


from matplotlib import gridspec  # make sure this import exists

def plot_cluster_scatters(clustering, pers_data, method, sample_count, max_clusters):
    cluster_ids = np.unique(clustering)

    # Nice name for the title
    method_name = str(method).split("(")[0] if not isinstance(method, str) else method

    # (Optional but helpful) order clusters by size: largest first, then id
    sizes = [(cid, np.sum(clustering == cid)) for cid in cluster_ids]
    sizes.sort(key=lambda x: (-x[1], x[0]))
    ordered_cluster_ids = [cid for cid, _ in sizes]

    # Limit the number of rows actually plotted
    rows = min(len(ordered_cluster_ids), max_clusters)
    cluster_ids_to_plot = ordered_cluster_ids[:rows]

    # Build figure for exactly `rows` rows
    fig = plt.figure(figsize=(6 * sample_count, 2 * rows))
    gs = gridspec.GridSpec(
        rows, sample_count + 1,
        width_ratios=[1] * sample_count + [0.05],
        wspace=0.15, hspace=0.2
    )

    axs = np.empty((rows, sample_count), dtype=object)
    scatter = None
    max_cluster_size = 0

    # Plot each selected cluster in its assigned row
    for row_i, cluster_id in enumerate(cluster_ids_to_plot):
        idx = np.where(clustering == cluster_id)[0]
        if idx.size == 0:
            continue

        size = min(sample_count, idx.size)
        np.random.seed(42)
        samples = np.random.choice(idx, size=size, replace=False)

        for j, sample in enumerate(samples):
            ax = fig.add_subplot(gs[row_i, j])
            axs[row_i, j] = ax
            scatter = pers_data.plot_raw_path(ax, sample, colour_bar=False)
            ax.set_title("")

            # y-label only in first column; show real cluster id
            ax.set_ylabel(f"Cluster {cluster_id}\nLatitude" if j == 0 else "")

            # x-label only for the deepest filled column so labels don't repeat
            ax.set_xlabel("Longitude" if j + 1 > max_cluster_size else "")

        max_cluster_size = max(size, max_cluster_size)

    # Colorbar in the final column (only if anything was plotted)
    if scatter is not None:
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label("Time Step")

    fig.text(0.5, 0.99, f"Paths of {method_name} clusters",
             ha='center', va='top', fontsize=20)

    # Keep your dynamic top margin, but use the actual number of rows
    title_space_inches = 0.4 + 0.02 * rows
    fig_height_inches = fig.get_figheight()
    top_margin = 1 - (title_space_inches / fig_height_inches)
    fig.subplots_adjust(top=top_margin)

    return fig



def plot_cluster_pers_diagrams(clustering, pers_data: PersistenceData, method, sample_count, max_clusters):
    cluster_ids = np.unique(clustering)
    method_name = str(method).split("(")[0] if not isinstance(method, str) else method

    # (Optional) order clusters by size: largest first, then id
    sizes = [(cid, np.sum(clustering == cid)) for cid in cluster_ids]
    sizes.sort(key=lambda x: (-x[1], x[0]))
    ordered_cluster_ids = [cid for cid, _ in sizes]

    # Limit number of rows actually plotted
    rows = min(len(ordered_cluster_ids), max_clusters)
    cluster_ids_to_plot = ordered_cluster_ids[:rows]

    # Build figure exactly for `rows` rows and `sample_count` columns
    fig = plt.figure(figsize=(6 * sample_count, 2 * rows))
    gs = gridspec.GridSpec(
        rows, sample_count,
        wspace=0.2, hspace=0.2
    )

    axs = np.empty((rows, sample_count), dtype=object)
    max_cluster_size = 0

    for row_i, cluster_id in enumerate(cluster_ids_to_plot):
        idx = np.where(clustering == cluster_id)[0]
        if idx.size == 0:
            continue

        size = min(sample_count, idx.size)
        np.random.seed(42)
        samples = np.random.choice(idx, size=size, replace=False)

        for j, sample in enumerate(samples):
            ax = fig.add_subplot(gs[row_i, j])
            axs[row_i, j] = ax
            pers_data.plot_persistence(ax, sample)
            ax.set_title("")
            ax.locator_params(axis='x', nbins=6)

            # y-label only in first column; show real cluster id
            ax.set_ylabel(f"Cluster {cluster_id}\nDeath" if j == 0 else "")

            # x-label only for the deepest filled column so labels don't repeat
            ax.set_xlabel("Birth" if j + 1 > max_cluster_size else "")

        max_cluster_size = max(size, max_cluster_size)

    fig.text(0.5, 0.99, f"Persistence diagrams of {method_name} clusters",
             ha='center', va='top', fontsize=20)

    # Dynamic top margin using the actual number of rows
    title_space_inches = 0.4 + 0.02 * rows
    fig_height_inches = fig.get_figheight()
    top_margin = 1 - (title_space_inches / fig_height_inches)
    fig.subplots_adjust(top=top_margin)

    return fig


def plot_scatter_matrix(data):
    size = data.shape[1]
    column_names = data.columns

    scatter_matrix(data, figsize=(size, size))
    plt.suptitle("Scatter Plot Matrix")
    plt.show()