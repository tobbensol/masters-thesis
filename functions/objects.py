import gudhi
import numpy as np
from gudhi.representations import Landscape
from matplotlib import pyplot as plt


class PersistenceData:
    def __init__(self, persistence, paths, resolution=10, num_landscapes=3):
        """
        Encapsulates persistence data, paths, statistics, and landscape transformation.

        :param persistence: List/array of persistence diagrams
        :param paths: List/array of paths corresponding to the persistence diagrams
        :param resolution: Resolution of the landscape transformation
        :param num_landscapes: Number of landscapes to compute
        """
        self.persistence = persistence
        self.paths = paths
        self.stats = self.compute_persistence_stats()
        self.landscapes = self.compute_landscapes(resolution, num_landscapes)
        self.resolution = resolution
        self.num_landscapes = num_landscapes

    def compute_persistence_stats(self):
        """Computes statistics for the persistence diagrams."""
        dataset = []
        for pers in self.persistence:
            births, deaths = pers[:, 0], pers[:, 1]
            lifespans = deaths - births

            data_row = [len(pers)]  # Number of persistence pairs
            lifespans.sort()
            top_5 = np.pad(lifespans[:5], (0, max(0, 5 - len(lifespans))), mode='constant')

            mean = np.mean(lifespans) if lifespans.size else 0
            median = np.median(lifespans) if lifespans.size else 0

            data_row.extend(list(top_5) + [mean, median])
            dataset.append(data_row)

        return np.array(dataset)

    def compute_landscapes(self, resolution, num_landscapes):
        """Computes landscape transformations for persistence diagrams."""
        landscape_model = Landscape(resolution=resolution, num_landscapes=num_landscapes)
        landscapes = [landscape_model.fit_transform([pers]) for pers in self.persistence]
        landscapes = np.array(landscapes)
        return landscapes.reshape(landscapes.shape[0], landscapes.shape[2])

    def plot_diagram(self, index=0):
        """Plots the persistence diagram and landscape of a given index."""
        fig, axs = plt.subplots(3, 1, figsize=(5, 15))

        path = self.paths[index]
        pers = self.persistence[index]
        landscape = self.landscapes[index]

        # Color points by index to create a gradient effect
        num_points = path.shape[0]
        colors = np.linspace(0, 1, num_points)  # Create a gradient scale

        scatter = axs[0].scatter(path[:, 0], path[:, 1], c=colors, cmap="plasma", edgecolors="none")
        axs[0].set_title("Flight Path (Gradient by Time)")
        axs[0].set_xlabel("Longitude")
        axs[0].set_ylabel("Latitude")

        # Add colorbar to indicate progression
        cbar = plt.colorbar(scatter, ax=axs[0])
        cbar.set_label("Time Step")

        gudhi.persistence_graphical_tools.plot_persistence_diagram(pers, axes=axs[1])
        axs[1].set_title("Persistence Diagram")

        for i in range(self.num_landscapes):
            axs[2].plot(landscape[i * self.resolution:(i + 1) * self.resolution], label=f"Landscape {i + 1}")

        axs[2].legend()
        axs[2].set_title("Persistence Landscape")
        plt.show()