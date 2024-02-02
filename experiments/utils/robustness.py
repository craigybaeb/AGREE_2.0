import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


class Robustness:
    def __init__(self):
        self.test = ""

    def linear_interpolate(self, query_sample, farthest_neighbor, num_steps=50):
        interpolated_samples = np.zeros((num_steps, query_sample.shape[1]))
        for step in range(num_steps):
            t = step / float(num_steps - 1)
            interpolated_samples[step] = query_sample + t * (farthest_neighbor - query_sample)
        return interpolated_samples
    
    
    def generate_perturbations(self, data_to_explain, model, X):
        perturbed_instances = []

        # Predict classes for the entire dataset
        y_pred = np.array(model.predict(X)).squeeze()
        y_pred_classes = y_pred.argmax(axis=1)

        for dte in data_to_explain:
            # Get the class of the instance to explain
            instance_class = model.predict(dte.reshape(1, -1)).argmax()

            # Separate instances by class, excluding the class of the instance to explain
            unlike_data = [X[i] for i in range(len(X)) if y_pred_classes[i] != instance_class]

            # Initialize NearestNeighbors for unlike data
            nun = NearestNeighbors(n_neighbors=len(unlike_data))
            nun.fit(unlike_data)

            # Find the nearest unlike neighbor
            _, nun_indices = nun.kneighbors(dte.reshape(1, -1))
            nun_idx = nun_indices[0][0]
            nun_instance = unlike_data[nun_idx]

            # Generate linear perturbations between the instance to explain and its nearest unlike neighbor
            perturbations = self.linear_interpolate(dte.reshape(1, -1), nun_instance.reshape(1, -1), 30)
            perturbed_instances.append(perturbations)

        return perturbed_instances

    def kendall_tau_distance(self, values1, values2):
        n = len(values1)
        assert len(values2) == n, "Both lists have to be of equal length"
        i, j = np.meshgrid(np.arange(n), np.arange(n))
        a = np.argsort(values1)
        b = np.argsort(values2)
        ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
        
        return ndisordered / (n * (n - 1))

    def calculate_robustness(self, original_explanation, perturbed_explanations, distance_method="euclidean", weight=0.7, show=None):
        calculate_distance = distance.euclidean

        if(distance_method == "cosine"):
            calculate_distance = distance.cosine
        elif(distance_method == "kendall"):
            calculate_distance = self.kendall_tau_distance

        distances = []
        for perturbed_explanation in perturbed_explanations:
            explanation_distance = calculate_distance(original_explanation, perturbed_explanation)
            distances.append(explanation_distance)


        distances = np.array(distances)

        if(weight > 0):
            # Calculate the weights using exponential decay based on the index
            weights = [np.exp(-weight * (i + 1)) for i in range(len(perturbed_explanations))]
            # Calculate the sum of the unnormalized weights
            sum_weights = sum(weights)

            # Normalize the weights
            normalized_weights = [w / sum_weights for w in weights]

            distances = np.array([distances[i] * normalized_weights[i] for i in range(len(distances))])

        # Calculate the minimum and maximum distances
        min_distance = np.min(distances)
        max_distance = np.max(distances)

        # Normalize the distances
        normalized_distances = []
        for d in distances: 
            if(max_distance > min_distance):
                normalized_distance = (d - min_distance) / (max_distance - min_distance)
            else:
                normalized_distance = 0
            normalized_distances.append(normalized_distance)

        normalized_distances = np.array(normalized_distances)

        # Calculate the similarity using 1 - normalized distance
        similarities = 1 - normalized_distances

        if(show == "similarity"):
            self.plot_robustness_curve(list(range(1,len(similarities)+1)), similarities, True)
        elif(show == "distance"):
            self.plot_robustness_curve(list(range(1,len(normalized_distances)+1)), normalized_distances, False)
        
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
    
    def generate_gaussian_perturbations(self, original_input, perturbation_radius):
        # This function adds small perturbations to the original input
        return original_input + np.random.uniform(-perturbation_radius, perturbation_radius, original_input.shape)

    def calculate_sensitivity(self, original_explanation, perturbed_explanations):
        max_difference = 0
        for perturbation in perturbed_explanations:
            distance = distance.euclidean(original_explanation, perturbation)
            max_difference = max(max_difference, distance)
        
        return max_difference


        

