import numpy as np
from sklearn.neighbors import NearestNeighbors

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

