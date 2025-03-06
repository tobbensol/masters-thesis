import gudhi
import numpy as np
from gudhi.representations import Landscape
from matplotlib import pyplot as plt


class PersistenceData:
    def __init__(self, persistence, paths, type, resolution=10, num_landscapes=3):
        self.persistence = persistence
        self.paths = paths
        self.stats = self.compute_persistence_stats()
        self.landscape_model = Landscape(resolution=resolution, num_landscapes=num_landscapes)
        self.landscapes = self.compute_landscapes()
        self.plot_text = {
            "LL": ["Flight Path", "Longitude", "Latitude"],
            "A" : ["Flight Altitude", "Timestep", "Altitude"],
            "S" : ["Flight Speed", "Timestep", "Speed"],
            "H" : ["Flight rotation", "Timestep", "Rotation(Radians)"],
        }[type]

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

    def compute_landscapes(self):
        """Computes landscape transformations for persistence diagrams."""
        landscapes = [self.landscape_model.fit_transform([pers]) for pers in self.persistence]
        landscapes = np.array(landscapes)
        return landscapes.reshape(landscapes.shape[0], landscapes.shape[2])

    def plot_diagram(self, index, axs=None, add_landscape:bool=False):
        if axs is None:
            fig_count = 3 if add_landscape else 2
            fig, axs = plt.subplots(fig_count, 1, figsize=(5, 5*fig_count))

        path = self.paths[index]
        pers = self.persistence[index]
        landscape = self.landscapes[index]

        num_points = path.shape[0]
        colors = np.linspace(0, 1, num_points)

        scatter = axs[0].scatter(path[:, 0], path[:, 1], c=colors, cmap="plasma", edgecolors="none")
        axs[0].set_title(self.plot_text[0])
        axs[0].set_xlabel(self.plot_text[1])
        axs[0].set_ylabel(self.plot_text[2])

        cbar = plt.colorbar(scatter, ax=axs[0])
        cbar.set_label("Time Step")

        gudhi.persistence_graphical_tools.plot_persistence_diagram(pers, axes=axs[1])
        axs[1].set_title("Persistence Diagram")

        if add_landscape:
            num_landscapes = self.landscape_model.num_landscapes
            resolution = self.landscape_model.resolution
            for i in range(num_landscapes):
                axs[2].plot(landscape[i * resolution:(i + 1) * resolution], label=f"Landscape {i + 1}")

            axs[2].legend()
            axs[2].set_title("Persistence Landscape")
