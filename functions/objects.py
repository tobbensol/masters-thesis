import gudhi
import numpy as np
from gudhi.representations import Landscape
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import svm


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

    @staticmethod
    def plot_multiple_diagrams(objects, index, landscapes: bool = False):
        """Plots multiple persistence diagrams and landscapes side by side."""
        num_objects = len(objects)
        object_fig_count = 3 if landscapes else 2

        fig, axs = plt.subplots(object_fig_count, num_objects,
                                figsize=(5 * num_objects, 5 * object_fig_count))  # 3 rows, N columns

        if num_objects == 1:
            axs = np.expand_dims(axs, axis=1)  # Ensure axs is always 2D (3 x N)

        for idx, obj in enumerate(objects):
            obj.plot_diagram(index, axs=axs[:, idx])  # Pass one column of subplots to each object

        plt.tight_layout()
        plt.show()


class Models:
    def __init__(self, seed):
        self.regressors = {
            "Support Vector Machines": [svm.SVR(), {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "poly"],
                "degree": [2, 3, 4],
            }],
            "Base line": [DummyRegressor(strategy="mean"), {

            }],
            "Multi-layer Perception": [MLPRegressor(random_state=seed, max_iter=5000), {
                "hidden_layer_sizes": [10, 20, 30],

            }],
            "K Nearest Neighbors": [KNeighborsRegressor(), {
                "n_neighbors": [5, 10, 20, 40],
                "p": [1, 2, 3]
            }],
            "Random Forrest Regressor": [RandomForestRegressor(random_state=seed), {
                "max_depth": [3, 6],
                "n_estimators": [25, 50, 100]
            }],
            "Decision Tree Regressor": [DecisionTreeRegressor(random_state=seed), {
                "min_samples_split": [2, 3, 4],
                "min_samples_leaf": [1, 2, 3]
            }],
        }

        self.classifiers = {
            "Support Vector Machines": [svm.SVC(random_state=seed), {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "poly"],
                "degree": [2, 3, 4],
            }],
            "Base line": [DummyClassifier(strategy="most_frequent"), {

            }],
            "Multi-layer Perception": [MLPClassifier(random_state=seed, max_iter=5000), {
                "hidden_layer_sizes": [10, 20, 30],

            }],
            "K Nearest Neighbors": [KNeighborsClassifier(), {
                "n_neighbors": [5, 10, 20, 40],
                "p": [1, 2, 3]
            }],
            "Random Forrest Regressor": [RandomForestClassifier(random_state=seed), {
                "max_depth": [3, 6],
                "n_estimators": [50, 100, 300]
            }],
            "Decision Tree Regressor": [DecisionTreeClassifier(random_state=seed), {
                "min_samples_split": [2, 3, 4],
                "min_samples_leaf": [1, 2, 3]
            }],
        }