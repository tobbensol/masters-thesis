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
            if len(pers) == 0:
                dataset.append([0] * 38)  # Placeholder for empty persistence diagram
                continue

            births, deaths = pers[:, 0], pers[:, 1]
            lifespans = deaths - births
            mid_points = (births + lifespans)/2

            count = len(pers)
            L_mu = np.sum(lifespans)
            if L_mu > 0:
                entropy = -np.sum(lifespans/L_mu * np.log2(lifespans/L_mu))
            else:
                entropy = 0

            data_row = [count, entropy]

            for i in [births, deaths, lifespans, mid_points]:
                mean = np.mean(i)
                std = np.std(i)
                median = np.median(i)
                range_val = np.max(i) - np.min(i)
                p10 = np.percentile(i, 10)
                p25 = np.percentile(i, 25)
                p75 = np.percentile(i, 75)
                p90 = np.percentile(i, 90)
                iqr = p75 - p25
                data_row.extend([mean, std, median, iqr, range_val, p10, p25, p75, p90])
            dataset.append(data_row)

        return np.array(dataset)

    def compute_landscapes(self):
        """Computes landscape transformations for persistence diagrams."""
        landscapes = [self.landscape_model.fit_transform([pers]) for pers in self.persistence]
        landscapes = np.array(landscapes)
        self.landscapes = landscapes.reshape(landscapes.shape[0], landscapes.shape[2])
        return self.landscapes

    def plot_raw_path(self, ax, index):
        """Plot the raw input trajectory on the given axis."""
        path = self.paths[index]

        if self.plot_text[1] != "Timestep":
            num_points = path.shape[0]
            colors = np.linspace(0, 1, num_points)
            scatter = ax.scatter(path[:, 1], path[:, 0], c=colors, cmap="plasma", edgecolors="none")
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Time Step")
        else:
            ax.scatter(path[:, 0], path[:, 1], cmap="plasma", edgecolors="none")

        ax.set_title(self.plot_text[0], fontsize=16)
        ax.set_xlabel(self.plot_text[1], fontsize=16)
        ax.set_ylabel(self.plot_text[2], fontsize=16)

    def plot_persistence(self, ax, index):
        """Plot the persistence diagram on the given axis."""
        pers = self.persistence[index]

        gudhi.persistence_graphical_tools.plot_persistence_diagram(pers, axes=ax)
        ax.set_title("Persistence Diagram", fontsize=16)

    def plot_landscape(self, ax, index):
        """Plot the persistence landscape on the given axis."""
        model = self.landscape_model
        landscape = self.landscapes[index]

        num_landscapes = model.num_landscapes
        resolution = model.resolution
        for i in range(num_landscapes):
            ax.plot(landscape[i * resolution:(i + 1) * resolution], label=f"Landscape {i + 1}")
        ax.legend()
        ax.set_title("Persistence Landscape", fontsize=16)
        ax.set_xlabel("Resolution steps", fontsize=16)

    def plot_diagram(self, index, axs=None, add_landscape: bool = False, save_svg=False):
        if axs is None:
            fig_count = 3 if add_landscape else 2
            fig, axs = plt.subplots(1, fig_count, figsize=(6 * fig_count, 5))
            fig.subplots_adjust(wspace=0.3)
        else:
            fig = plt.gcf()

        self.plot_raw_path(axs[0], index)
        self.plot_persistence(axs[1], index)

        if add_landscape:
            self.plot_landscape(axs[2], index)

        return fig


    @staticmethod
    def plot_multiple_diagrams(objects, index, add_landscape:bool=False):
        """Plots multiple persistence diagrams and landscapes side by side."""
        num_objects = len(objects)
        object_fig_count = 3 if add_landscape else 2

        fig, axs = plt.subplots(num_objects, object_fig_count,
                                figsize=(5 * object_fig_count, 5 * num_objects))  # 3 rows, N columns

        if num_objects == 1:
            axs = np.expand_dims(axs, axis=1)  # Ensure axs is always 2D (3 x N)

        for idx, obj in enumerate(objects):
            obj.plot_diagram(index, axs=axs[idx, :], add_landscape=add_landscape)  # Pass one column of subplots to each object

        plt.tight_layout()
        plt.show()


class Models:
    def __init__(self, seed):
        self.regressors = {
            "Base line": [DummyRegressor(strategy="mean"), {}],  # Fine as is

            "Support Vector Machines": [svm.SVR(), {
                "C": [0.1, 1, 10, 100],  # Regularization strength
                "degree": [2, 3, 4],  # Only used for "poly" kernel
                "gamma": ["scale", "auto"],  # Kernel coefficient
            }],

            "Multi-layer Perceptron": [MLPRegressor(random_state=seed, max_iter=5000), {
                "hidden_layer_sizes": [(10,), (25,), (25, 25), (10, 10)],  # Different layer structures
                "alpha": [0.0001, 0.001, 0.01],  # L2 regularization
                "learning_rate": ["constant", "adaptive"],  # Learning rate strategy
            }],

            "K Nearest Neighbors": [KNeighborsRegressor(), {
                "n_neighbors": [5, 10, 25, 50],  # Range of neighbors
                "weights": ["uniform", "distance"],  # Weighting scheme
            }],

            "Random Forest Regressor": [RandomForestRegressor(random_state=seed), {
                "n_estimators": [25, 50, 150],  # More trees for better stability
                "max_depth": [5, 10, 20],  # None allows full depth
            }],

            "Decision Tree Regressor": [DecisionTreeRegressor(random_state=seed), {
                "max_depth": [5, 10, 20],  # Depth of the tree
                "min_samples_split": [2, 3, 4],
                "min_samples_leaf": [1, 2, 3]
            }]
        }

        self.classifiers = {
            "Base line": [DummyClassifier(strategy="most_frequent"), {}],  # Fine as is

            "Support Vector Machines": [svm.SVC(random_state=seed, probability=True, class_weight="balanced"), {
                "C": [0.1, 1, 10, 100],  # Regularization strength
                "degree": [2, 3, 4],  # Only used for "poly" kernel
                "gamma": ["scale", "auto"],  # Kernel coefficient
            }],

            "Multi-layer Perceptron": [MLPClassifier(random_state=seed, max_iter=5000), {
                "hidden_layer_sizes": [(10,), (25,), (25, 25), (10, 10)],  # Different layer structures
                "alpha": [0.001, 0.01],  # L2 regularization
                "learning_rate": ["constant", "adaptive"],  # Learning rate strategy
            }],

            "K Nearest Neighbors": [KNeighborsClassifier(), {
                "n_neighbors": [5, 15, 50],  # Range of neighbors
                "weights": ["uniform", "distance"],  # Weighting scheme
            }],

            "Random Forest Classifier": [RandomForestClassifier(random_state=seed, class_weight="balanced"), {
                "n_estimators": [25, 50, 150],  # More trees for better stability
                "max_depth": [5, 10, 20],  # None allows full depth
            }],

            "Decision Tree Classifier": [DecisionTreeClassifier(random_state=seed, class_weight="balanced"), {
                "max_depth": [5, 10, 20],  # Depth of the tree
                "min_samples_split": [2, 3, 4],
            }]
        }