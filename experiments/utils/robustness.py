import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import matplotlib.pyplot as plt

class Robustness:
    def __init__(self):
        self.test = ""

    def linear_interpolate(self, query_sample, farthest_neighbor, num_steps=50):
        interpolated_samples = np.zeros((num_steps, query_sample.shape[1]))
        for step in range(1, num_steps):
            t = step / float(num_steps - 1)
            interpolated_samples[step] = query_sample + t * (farthest_neighbor - query_sample)
        return interpolated_samples
    
    def generate_perturbations(self, data_to_explain, model, X):
        perturbed_instances = []

        for dte in data_to_explain:

            y_pred = np.array(model.predict(X)).squeeze()
            y_pred_binary = y_pred.argmax(axis=1)

            # Step 7: Filter instances belonging to each class separately
            class_0_instances = []
            class_1_instances = []

            for i in range(len(y_pred_binary)):
                if y_pred_binary[i] == 0:
                    class_0_instances.append(X[i])
                else:
                    class_1_instances.append(X[i])

            # Select a sample from the test set
            instance_to_explain = dte

            instance_class = model.predict(instance_to_explain.reshape(1,-1)).argmax()
            like_data = []
            unlike_data = []

            if(instance_class == 0):
                like_data = class_0_instances
                unlike_data = class_1_instances
            else:
                like_data = class_1_instances
                unlike_data = class_0_instances

            # Initialize NearestNeighbors
            nn = NearestNeighbors(n_neighbors=len(like_data))
            nun = NearestNeighbors(n_neighbors=len(unlike_data))

            # Fit on training data
            nn.fit(like_data)
            nun.fit(unlike_data)

            # Find 5 nearest neighbors
            distances, indices = nn.kneighbors(instance_to_explain.reshape(1,-1))
            _, nun_indices = nun.kneighbors(instance_to_explain.reshape(1,-1))

            distances = distances[0][1:]
            indices = indices[0][1:]

            # The farthest of the 5 nearest neighbors
            farthest_neighbor_idx = indices
            farthest_neighbors = [like_data[i] for i in farthest_neighbor_idx]

            nun_idx = nun_indices[0][0]
            nun_instance = unlike_data[nun_idx]

            nn_nun = NearestNeighbors(n_neighbors=len(like_data) - 2)

            nn_nun.fit(farthest_neighbors)

            nn_nun_distances, nn_nun_indices = nn_nun.kneighbors(nun_instance.reshape(1,-1))

            nn_nun_distances = nn_nun_distances[0][1:]
            nn_nun_indices = nn_nun_indices[0][1:]

            nn_nun_idx = nn_nun_indices[0]
            nn_nun_instance = farthest_neighbors[nn_nun_idx]

            perturbations = self.linear_interpolate(dte.reshape(1,-1), nn_nun_instance, 30)
            perturbed_instances.append(perturbations)
        
        return perturbed_instances

    def calculate_robustness(self, original_explanation, perturbed_explanations, distance_method="euclidean", weight=0.7):
        calculate_distance = distance.euclidean

        if(distance_method == "cosine"):
            calculate_distance = distance.cosine

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
        normalized_distances = (distances - min_distance) / (max_distance - min_distance)

        # Calculate the similarity using 1 - normalized distance
        similarities = 1 - normalized_distances

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


        

