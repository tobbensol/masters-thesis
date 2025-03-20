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

            # Sort indices by lifespans in descending order
            sorted_indices = np.argsort(lifespans)[::-1]

            # Apply sorting to births, deaths, and lifespans
            sorted_births = births[sorted_indices]
            sorted_deaths = deaths[sorted_indices]
            sorted_lifespans = lifespans[sorted_indices]

            # Get top 5 elements, padded if necessary
            top_5_births = np.pad(sorted_births[:5], (0, max(0, 5 - len(sorted_births))), mode='constant')
            top_5_deaths = np.pad(sorted_deaths[:5], (0, max(0, 5 - len(sorted_deaths))), mode='constant')
            top_5_lifespans = np.pad(sorted_lifespans[:5], (0, max(0, 5 - len(sorted_lifespans))), mode='constant')

            # Compute statistics
            mean_lifespan = np.mean(lifespans) if lifespans.size else 0
            median_lifespan = np.median(lifespans) if lifespans.size else 0

            # Create data row with number of persistence pairs, top 5 values, mean, and median
            data_row = [len(pers), mean_lifespan, median_lifespan] + list(top_5_lifespans) + list(top_5_births) + list(top_5_deaths)
            dataset.append(data_row)

        return np.array(dataset)

    def compute_landscapes(self):
        """Computes landscape transformations for persistence diagrams."""
        landscapes = [self.landscape_model.fit_transform([pers]) for pers in self.persistence]
        landscapes = np.array(landscapes)
        self.landscapes = landscapes.reshape(landscapes.shape[0], landscapes.shape[2])
        return self.landscapes

    def plot_diagram(self, index, axs=None, add_landscape:bool=False, save_svg=False):
        if axs is None:
            fig_count = 3 if add_landscape else 2
            fig, axs = plt.subplots(1, fig_count, figsize=(6*fig_count, 5))
            fig.subplots_adjust(wspace=0.3)

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
            "Support Vector Machines": [svm.SVR(), {
                "C": [0.1, 1, 10, 100],  # Regularization strength
                "degree": [2, 3, 4],  # Only used for "poly" kernel
                "gamma": ["scale", "auto"],  # Kernel coefficient
            }],

            "Base line": [DummyRegressor(strategy="mean"), {}],  # Fine as is

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
            "Support Vector Machines": [svm.SVC(random_state=seed, probability=True, class_weight="balanced"), {
                "C": [0.1, 1, 10, 100],  # Regularization strength
                "degree": [2, 3, 4],  # Only used for "poly" kernel
                "gamma": ["scale", "auto"],  # Kernel coefficient
            }],

            "Base line": [DummyClassifier(strategy="most_frequent"), {}],  # Fine as is

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