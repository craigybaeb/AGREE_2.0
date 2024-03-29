import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import math


class Robustness:
    def __init__(self):
        self.test = ""

    def linear_interpolate(self, query_sample, farthest_neighbor, num_steps=30, categorical_columns=[]):
        # Initialize an array to hold the interpolated samples
        interpolated_samples = np.zeros((num_steps, len(query_sample)))

        # Handle numerical interpolation
        for step in range(num_steps):
            t = step / float(num_steps - 1)
            interpolated_samples[step, :] = query_sample + t * (farthest_neighbor - query_sample)

        # Handle categorical columns
        for cat_col in categorical_columns:
            midpoint = num_steps // 2
            for step in range(num_steps):
                if step < midpoint:
                    interpolated_samples[step, cat_col] = query_sample[cat_col]
                else:
                    interpolated_samples[step, cat_col] = farthest_neighbor[cat_col]

        return interpolated_samples
    
    
    def generate_perturbations(self, data_to_explain, model, X, num_perturbations = 30, categorical_columns = []):
        perturbed_instances = []

        # Predict classes for the entire dataset
        y_pred = np.array(model.predict(X)).squeeze()
        y_pred_classes = y_pred.argmax(axis=1)

        for dte in data_to_explain:
            # Get the class of the instance to explain
            instance_class = model.predict(dte.reshape(1, -1)).argmax()

            nun_instance = self.get_nearest_unlike_neighbour(instance_class, dte, X, y_pred_classes)
            fln_instance = self.get_farthest_like_neighbour(instance_class, nun_instance, dte, X, y_pred_classes)

            # Generate linear perturbations between the instance to explain and its nearest unlike neighbor
            perturbations = self.linear_interpolate(dte, fln_instance, num_perturbations, categorical_columns)
            perturbed_instances.append(perturbations)

        return perturbed_instances

    def get_farthest_like_neighbour(self, instance_class, nearest_unlike_neighbour, data_to_explain, X, classes):
        like_data = [X[i] for i in range(len(X)) if classes[i] == instance_class]

        # Initialize NearestNeighbors for like data
        nn = NearestNeighbors(n_neighbors=len(like_data))
        nn.fit(like_data)

        fln_distances, fln_indices = nn.kneighbors(nearest_unlike_neighbour.reshape(1,-1))

        fln_distances = fln_distances[0][1:]
        fln_indices = fln_indices[0][1:]

        fln_idx = fln_indices[0]
        fln_instance = np.array(like_data[fln_idx])

        idx = 1 # Query the same so enter while loop
        while(np.array_equal(fln_instance, data_to_explain)):
            fln_idx = fln_indices[idx] # get next instance
            idx += 1 # Continue getting next instance if the same
            fln_instance = np.array(like_data[fln_idx])

        return fln_instance
        
    def get_nearest_unlike_neighbour(self, instance_class, data_to_explain, X, classes):
        # Separate instances by class, excluding the class of the instance to explain
        unlike_data = [X[i] for i in range(len(X)) if classes[i] != instance_class]

        # Initialize NearestNeighbors for unlike data
        nun = NearestNeighbors(n_neighbors=len(unlike_data))
        nun.fit(unlike_data)

        # Find the nearest unlike neighbor
        _, nun_indices = nun.kneighbors(data_to_explain.reshape(1, -1))
        nun_idx = nun_indices[0][1] #Replace with 1????
        nun_instance = unlike_data[nun_idx]

        return nun_instance

    def kendall_tau_distance(self, values1, values2):
        n = len(values1)
        assert len(values2) == n, "Both lists have to be of equal length"
        i, j = np.meshgrid(np.arange(n), np.arange(n))
        a = np.argsort(values1)
        b = np.argsort(values2)
        ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
        
        return ndisordered / (n * (n - 1))

    def calculate_robustness(self, original_explanation, perturbed_explanations, r=2, categorical_columns = [], ranges = [], show=None):

        distances = []
        for perturbed_explanation in perturbed_explanations:
            explanation_distance = self.calculate_gower_distance(original_explanation, perturbed_explanation, r, categorical_columns, ranges)
            distances.append(explanation_distance)

        distances = np.array(distances)

        # Calculate the similarity using 1 - normalized distance
        similarities = 1 - distances

        if(show == "similarity"):
            self.plot_robustness_curve(list(range(1,len(similarities)+1)), similarities, True)
        elif(show == "distance"):
            self.plot_robustness_curve(list(range(1,len(distances)+1)), distances, False)
        
        # Create an array of x-values
        x_values = np.linspace(0, 1, len(similarities))

        # Calculate AUC using the trapezoidal rule
        auc = np.trapz(similarities, x=x_values)

        return auc
    
    def plot_robustness_curve(self, x, y, similarity=True):
        plt.plot(x, y)
        plt.xlabel("Perturbation")
        if(similarity):
            plt.ylabel("Similarity")
        else:
            plt.ylabel("Distance")
        plt.show()

        return
    
    def generate_gaussian_perturbations(self, original_input, perturbation_radius, categorical_features = []):
        # This function adds small perturbations to the original input
        perturbation = original_input + np.random.uniform(-perturbation_radius, perturbation_radius, original_input.shape)

        # Reset categorical features to original to ensure in distribution
        for cat in categorical_features:
            perturbation[cat] = original_input[cat]

        return perturbation

    def calculate_sensitivity(self, original_explanation, perturbed_explanations):
        max_difference = 0
        for perturbation in perturbed_explanations:
            d = distance.euclidean(original_explanation, perturbation)
            max_difference = max(max_difference, d)
        
        return max_difference
    
    def calculate_gower_distance(self, query, perturbation, r, categorical_columns = [], ranges = []):

        distance = 0
        for i, feature in enumerate(query):

            if(i in categorical_columns):
                if(feature != perturbation[i]):
                    distance += 1
            else:
                numeric_distance = pow(abs(feature - perturbation[i]), r) / ranges[i]
                distance += numeric_distance
        return distance / len(query)