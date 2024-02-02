# Explainers
import shap
import lime
import innvestigate # Smoothgrad, Vanilla Gradients, Input x Gradients, Layerwise Relevance Propagation, Guided Backpropagation, Deep Taylor
from shap import DeepExplainer, GradientExplainer, KernelExplainer
from shap.explainers import Permutation
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils.robustness import Robustness

class Explanation():
    def __init__(self, feature_names, class_labels, explainers_to_use, num_features, filepath, include_gaussian=False):
        self.feature_names = feature_names
        self.class_labels = class_labels
        self.explainers_to_use = explainers_to_use
        self.num_features = num_features
        self.filepath = filepath
        self.include_gaussian = include_gaussian

        self.robustness = Robustness()
        

    def get_predict_function(self, model):
        # We need to define a function to turn the probabilities into predictions for Kernel SHAP
        def predict_function(X):

            # Get the probabilites from the model
            probabilities_y = model.predict(X)

            # Convert the probabilities to predictions
            predictions = []

            for probability in probabilities_y:
                predictions.append(probability.argmax(axis=-1))

            return np.array(predictions)
        return predict_function
    
    def display_feature_importance(self, feature_names, feature_importance, filepath, explainer, fold_num, prediction, perturbed=False, show=False):
        # Create a DataFrame to plot an example local Smoothgrad explanation
        df = pd.DataFrame({
            "Features": feature_names,
            "Feature Importance": feature_importance
                })

        # Sort the DataFrame to easily see most important features
        df = df.reindex(df['Feature Importance'].abs().sort_values(ascending=False).index)

        # Plot the DataFrame
        sns.barplot(y='Features', x='Feature Importance', data=df)
        if(perturbed):
            plt.title(f"Example {explainer} perturbed explanation (Instance {prediction}, Fold {fold_num})")
            if(show):
                plt.show()
            plt.savefig(f'{filepath}/explanations/figures/perturbed_feature_importance_{explainer}_fold_{fold_num}.png')
        else:
            plt.title(f"Example {explainer} explanation (Instance {prediction}, Fold {fold_num})")
            if(show):
                plt.show()
            plt.savefig(f'{filepath}/explanations/figures/feature_importance_{explainer}_fold_{fold_num}.png')

        return
    
    def display_gradients(self, example, filepath, explainer, fold_num, prediction, perturbed=False, show=False):
        perturbed_tag = ""
        perturbed_path = ""
        if(perturbed):
            perturbed_tag = "perturbed"
            perturbed_path = "perturbed_"

        if(show):
            plt.show()
        plt.title(f"Visualisation of {perturbed_tag} gradients using {explainer} (Instance {prediction}, Fold {fold_num})")
        if(show):
            plt.show()
        plt.savefig(f'{filepath}/explanations/figures/{perturbed_path}gradient_explanations{explainer}_fold_{fold_num}.png')

        return
    
    def get_kernel_shap_explanations(self, background_data, data_to_explain, predict_function, feature_names, filepath, fold_num, prediction, perturbed=False, show=True):
        # Get the Kernel SHAP explanations
        kernel_shap = KernelExplainer(predict_function, background_data[:10]) # A single background sample is enough, but we will use 10 for better performance
        kernel_shap_values = kernel_shap.shap_values(data_to_explain)

        if(show):
            self.display_feature_importance(feature_names, kernel_shap_values[prediction], filepath, "Kernel SHAP", fold_num, prediction, perturbed)

        return kernel_shap_values
    
    def get_maple_explanations(self, background_data, data_to_explain, predict_function, feature_names, filepath, fold_num, prediction, perturbed=False, show=True):
        # Get the MAPLE explanations

        # Instantiate the MAPLE explainer
        maple_explainer = shap.explainers.other.Maple(predict_function, background_data[:1000]) # 1000 background samples should be sufficient

        # Get the MAPLE explanations
        maple_explanations = maple_explainer.attributions(data_to_explain)

        if(show):
            self.display_feature_importance(feature_names, maple_explanations[prediction], filepath, "MAPLE", fold_num, prediction, perturbed)

        return maple_explanations
    
    def get_lime_explanations(self, background_data, data_to_explain, model, feature_names, class_labels, filepath, fold_num, prediction, perturbed=False, show=True):
        # Instatiate the LIME explanation instance
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            background_data,
            feature_names = feature_names,
            discretize_continuous = False,
            class_names = class_labels,
            mode='classification')

        # Explain an instance using LIME
        lime_explanation = lime_explainer.explain_instance(
            data_row = np.squeeze(np.array(data_to_explain[0:1])),
            predict_fn = model.predict,
            num_features=self.num_features)

        if(show):
            # Display an example LIME explanation for an
            perturbed_tag = ""
            if(perturbed):
                perturbed_tag = "perturbed_"
            lime_explanation.as_pyplot_figure()
            # plt.show()
            plt.savefig(f'{filepath}/explanations/figures/{perturbed_tag}lime_fold_{fold_num}.png')

            lime_explanation.show_in_notebook(show_table=True)
            lime_explanation.save_to_file(f"{filepath}/explanations/figures/lime_fold_{fold_num}.html")

        lime_explanations = []

        for i in range(len(data_to_explain)):

            # Explain the instance using LIME
            lime_explanation = lime_explainer.explain_instance(
            data_row = data_to_explain[i],
            predict_fn = model.predict,
            num_features = self.num_features).as_map()

            # Extract the explanation from the explanation object
            sort = sorted(lime_explanation[1], key=lambda explanation_tuple: explanation_tuple[0])
            mapped = list(map(lambda x: x[1], sort))

            lime_explanations.append(mapped)

        return lime_explanations
    
    def get_gradient_shap_explanations(self, background_data, data_to_explain, predictions, model, feature_names, filepath, fold_num, prediction, perturbed=False, show=True):

        # Get the Gradient SHAP explanations
        gradient_shap = GradientExplainer(model, [background_data])
        gradient_shap_values = gradient_shap.shap_values(data_to_explain)

        # Extract the feature importances for the predicted class
        gradient_shap_classwise_values = []

        for i in range(len(data_to_explain)):
            gradient_shap_classwise_values.append(gradient_shap_values[predictions[i]][i])

        if(show):
            self.display_feature_importance(feature_names, gradient_shap_classwise_values[prediction], filepath, "Gradient SHAP", fold_num, prediction, perturbed)

        return gradient_shap_classwise_values
    
    def get_deep_shap_explanations(self, background_data, data_to_explain, predictions, model, feature_names, filepath, fold_num, prediction, perturbed=False, show=True):
        # Get the Deep SHAP explanations
        deep_shap = DeepExplainer(model, background_data[:1000]) # 1000 background samples should be enough
        deep_shap_values = deep_shap.shap_values(data_to_explain)

        # Extract the feature importances for the predicted class
        deep_shap_classwise_values = []

        for i in range(len(data_to_explain)):
            deep_shap_classwise_values.append(deep_shap_values[predictions[i]][i])

        if(show):
            self.display_feature_importance(feature_names, deep_shap_classwise_values[prediction], filepath, "Deep SHAP", fold_num, prediction, perturbed)

        return deep_shap_classwise_values
    
    def get_smoothgrad_explanations(self, data_to_explain, model_without_softmax, feature_names, filepath, fold_num, prediction, perturbed=False, show=True):
        # Define some noise to use in Smoothgrad
        input_range = [-1, 1]
        noise_scale = (input_range[1] - input_range[0]) * 0.1

        # Creating a Smoothgrad analyzer
        gradient_analyzer = innvestigate.create_analyzer(
            "smoothgrad",
            model_without_softmax,
            postprocess = "square",
            noise_scale=noise_scale
            )

        # Applying the analyzer
        smoothgrad = gradient_analyzer.analyze(data_to_explain)

        # Displaying the gradients
        if(show):
            example_flattened = smoothgrad[prediction].reshape(1, self.num_features, 1)
            self.display_gradients(example_flattened, filepath, "Smoothgrad", fold_num, prediction, perturbed)

            # Plot an example of a local Smoothgrad explanation
            example = smoothgrad[prediction].reshape(self.num_features)
            self.display_feature_importance(feature_names, example, filepath, "Smoothgrad", fold_num, prediction, perturbed)

        return smoothgrad
    
    def get_vanilla_gradients_explanations(self, data_to_explain, model_without_softmax, feature_names, filepath, fold_num, prediction, perturbed=False, show=True):
        # Creating a Vanilla Gradients analyzer
        gradient_analyzer = innvestigate.create_analyzer("gradient", model_without_softmax)

        # Applying the analyzer
        vanilla_gradients = gradient_analyzer.analyze(data_to_explain)

        if(show):
            # Displaying the gradients
            example_flattened = vanilla_gradients[prediction].reshape(1, self.num_features, 1)
            self.display_gradients(example_flattened, filepath, "Vanilla Gradients", fold_num, prediction, perturbed)

            # Plot an example of a local Vanilla Gradients explanation
            example = vanilla_gradients[prediction].reshape(self.num_features)
            self.display_feature_importance(feature_names, example, filepath, "Vanilla Gradients", fold_num, prediction, perturbed)

        return vanilla_gradients
    
    def get_gradients_x_input_explanations(self, data_to_explain, model_without_softmax, feature_names, filepath, fold_num, prediction, perturbed=False, show=True):
        # Creating an Gradients x Input analyzer
        gradient_analyzer = innvestigate.create_analyzer("input_t_gradient", model_without_softmax)

        # Applying the analyzer
        gradients_x_input = gradient_analyzer.analyze(data_to_explain)

        if(show):
            # Displaying the gradients
            example_flattened = gradients_x_input[prediction].reshape(1, self.num_features, 1)
            self.display_gradients(example_flattened, filepath, "Gradients x Input", fold_num, prediction, perturbed)

            # Plot an example of a local Gradients X Input explanation
            example = gradients_x_input[prediction].reshape(self.num_features)
            self.display_feature_importance(feature_names, example, filepath, "Gradients x Input", fold_num, prediction, perturbed)

        return gradients_x_input
    
    def get_guided_backpropagation_explanations(self, data_to_explain, model_without_softmax, feature_names, filepath, fold_num, prediction, perturbed=False, show=True):
        # Creating a Guided Backpropagation analyzer
        gradient_analyzer = innvestigate.create_analyzer("guided_backprop", model_without_softmax)

        # Applying the analyzer
        guided_backpropagation = gradient_analyzer.analyze(data_to_explain)

        if(show):
            # Displaying the gradients
            example_flattened = guided_backpropagation[prediction].reshape(1, self.num_features, 1)
            self.display_gradients(example_flattened, filepath, "Guided Backpropagation", fold_num, prediction, perturbed)

            # Plot an example of a local Guided Backpropagation explanation
            example = guided_backpropagation[prediction].reshape(self.num_features)
            self.display_feature_importance(feature_names, example, filepath, "Guided Backpropagation", fold_num, prediction, perturbed)

        return guided_backpropagation
    
    def get_layerwise_relevance_propagation_explanations(self, data_to_explain, model_without_softmax, feature_names, filepath, fold_num, prediction, rule="lrp.epsilon", perturbed=False, show=True):
        # Creating a Layerwise Relevance Propagation analyzer
        gradient_analyzer = innvestigate.create_analyzer(rule, model_without_softmax)

        # Applying the analyzer
        layerwise_relevance_propagation = gradient_analyzer.analyze(data_to_explain)

        if(show):
            # Displaying the gradients
            example_flattened = layerwise_relevance_propagation[prediction].reshape(1, self.num_features, 1)
            self.display_gradients(example_flattened, filepath, "Layerwise Relevance Propagation", fold_num, prediction, perturbed)

            # Plot an example of a local Layerwise Relevance Propagation explanation
            example = layerwise_relevance_propagation[prediction].reshape(self.num_features)
            self.display_feature_importance(feature_names, example, filepath, "Layerwise Relevance Propagation", fold_num, prediction, perturbed)

        return layerwise_relevance_propagation
    
    def get_deep_taylor_explanations(self, data_to_explain, model_without_softmax, feature_names, filepath, fold_num, prediction, perturbed=False, show=True):
        # Creating a Deep Taylor analyzer
        gradient_analyzer = innvestigate.create_analyzer("deep_taylor", model_without_softmax)

        # Applying the analyzer
        deep_taylor = gradient_analyzer.analyze(data_to_explain)

        if(show):
            # Displaying the gradients
            example_flattened = deep_taylor[prediction].reshape(1, self.num_features, 1)
            self.display_gradients(example_flattened, filepath, "Deep Taylor Decomposition", fold_num, prediction, perturbed)

            # Plot an example of a local Deep Taylor explanation
            example = deep_taylor[prediction].reshape(self.num_features)
            self.display_feature_importance(feature_names, example, filepath, "Deep Taylor Decomposition", fold_num, prediction, perturbed)

        return deep_taylor
    
    def get_integrated_gradients_explanations(self, data_to_explain, model_without_softmax, feature_names, filepath, fold_num, prediction, baseline="zero", num_iterations=10, perturbed=False, show=True):

        # Initialise the Integrated Gradients explanation
        integrated_gradients = None

        if(baseline == "zero"):
            # Creating an Integrated Gradients analyzer with zero baseline
            gradient_analyzer = innvestigate.create_analyzer("integrated_gradients", model_without_softmax)

            # Applying the analyzer
            integrated_gradients = gradient_analyzer.analyze(data_to_explain)
        elif(baseline == "uniform"):
            # Initialise the results from different uniform baselines
            integrated_gradients_uniform_repetitions = []

            # We loop over several uniform baselines and average the results
            for i in range(num_iterations):
                # Generate a uniform baseline
                uniform_baseline = np.random.rand(self.num_features)

                # Creating an Integrated Gradients analyzer
                gradient_analyzer = innvestigate.create_analyzer("integrated_gradients", model_without_softmax, reference_inputs=uniform_baseline)

                # Applying the analyzer
                integrated_gradients_uniform = gradient_analyzer.analyze(data_to_explain)
                integrated_gradients_uniform_repetitions.append(integrated_gradients_uniform)

                # Average all of the Integrated Gradients explanations for all uniform baselines
                integrated_gradients =  np.mean(integrated_gradients_uniform_repetitions, axis=0)

        if(show):
            # Displaying the gradients
            example_flattened = integrated_gradients[prediction].reshape(1, self.num_features, 1)
            self.display_gradients(example_flattened, filepath, "Integrated Gradients", fold_num, prediction, perturbed)

            # Plot an example of a local Integrated Gradients explanation
            example = integrated_gradients[prediction].reshape(self.num_features)
            self.display_feature_importance(feature_names, example, filepath, "Integrated Gradients", fold_num, prediction, perturbed)

        return integrated_gradients
    
    def save_explanation_state(self, data_to_save, filepath):
        with open(f'{filepath}/explanations/results/explanation_state.pkl', 'wb') as file:
            pickle.dump(data_to_save, file)
        print(f"Data saved to '{filepath}/explanations/results/explanation_state.pkl'")

    def get_explanations(self, X, y, seeds, filepath, repeat_predictions, checkpoint, num_explanations=50, resume=False, example_index=0):
        explanations_outer = []
        perturbed_explanations_outer = []
        gaussian_perturbed_explanations_outer = []
        random_samples_outer = []

        for i, seed in enumerate(seeds):

            if(checkpoint > i and resume):
                print(f"Skipping split {i} as checkpoint exists in later iteration.")
                continue

            print(f"Starting fold {i}...")

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)

            model = tf.keras.models.load_model(f'{filepath}/training/models/model_{i}.h5')

            # Strip Softmax layer from model to enable the remainder explainer methods
            model_without_softmax = innvestigate.model_wo_softmax(model)

            predict_function = self.get_predict_function(model)

            predictions = repeat_predictions[i]

            example_instance_to_explain = example_index

            # Filter out instances where the model's prediction matches the true label
            correctly_predicted_indices = [i for i in range(len(y_test)) if np.array(y_test)[i] == predictions[i]]

            # Create new arrays with only the correct instances
            X_test_correct = np.array(X_test)[correctly_predicted_indices]
            predicted_y_correct = np.array(predictions)[correctly_predicted_indices]

            data_to_explain = X_test_correct

            random_samples = []

            if(len(data_to_explain) > num_explanations):
                # Get the number of rows in the 2D array
                num_rows = data_to_explain.shape[0]
                
                np.random.seed(42)
                # Generate 1000 random row indices
                random_row_indices = np.random.choice(num_rows, num_explanations, replace=False)

                random_samples = random_row_indices

                # Use the random row indices to select 1000 instances from the 2D array
                data_to_explain = data_to_explain[random_row_indices, :]
                predicted_y_correct[random_row_indices]

            random_samples_outer.append(random_samples)

            perturbations = self.robustness.generate_perturbations(data_to_explain, model, X)
            
            gaussian_perturbations = []

            if(self.include_gaussian == True):
                for instance in data_to_explain:
                    i_gaussian_perturbations = []
                    for i in range(30):
                        i_gaussian_perturbations.append(self.robustness.generate_gaussian_perturbations(instance, 0.01))
                    gaussian_perturbations.append(i_gaussian_perturbations)
                    
            perturbed_predictions = []
            for i, _ in enumerate(data_to_explain):
                instance_perturbed_predictions = model.predict(perturbations[i])
                instance_predictions = np.array(instance_perturbed_predictions).argmax(axis=1)
                perturbed_predictions.append(instance_predictions)

            gaussian_perturbed_predictions = []
            for i, _ in enumerate(data_to_explain):
                gaussian_instance_perturbed_predictions = model.predict(gaussian_perturbations[i])
                gaussian_instance_predictions = np.array(gaussian_instance_perturbed_predictions).argmax(axis=1)
                gaussian_perturbed_predictions.append(gaussian_instance_predictions)

            # Create dictionaries to store the explanations
            explanations = {explainer: [] for explainer in self.explainers_to_use}
            perturbed_explanations = {explainer: [] for explainer in self.explainers_to_use}
            gaussian_perturbed_explanations = {explainer: [] for explainer in self.explainers_to_use}

            for explainer in self.explainers_to_use:
                if(explainer == "deep_shap"):
                    print('Getting Deep SHAP explanations...')
                    explanations[explainer] = self.get_deep_shap_explanations(X_train, data_to_explain, predicted_y_correct, model, self.feature_names, filepath, i, example_instance_to_explain)
                    for j, instance in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_deep_shap_explanations(X_train, perturbations[j], perturbed_predictions[j], model, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_deep_shap_explanations(X_train, gaussian_perturbations[j], gaussian_perturbed_predictions[j], model, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)

                elif(explainer == "gradient_shap"):
                    print('Getting Gradient SHAP explanations...')
                    explanations[explainer] = self.get_gradient_shap_explanations(X_train, data_to_explain, predicted_y_correct, model, self.feature_names, filepath, i, example_instance_to_explain)
                    for j, instance in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_gradient_shap_explanations(X_train, perturbations[j], perturbed_predictions[j], model, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_gradient_shap_explanations(X_train, gaussian_perturbations[j], gaussian_perturbed_predictions[j], model, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)
                elif(explainer == "lime"):
                    print('Getting LIME explanations...')
                    explanations[explainer] = self.get_lime_explanations(X_train, data_to_explain, model, self.feature_names, self.class_labels, filepath, i, example_instance_to_explain)
                    for j, instance in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_lime_explanations(X_train, perturbations[j], model, self.feature_names, self.class_labels, filepath, i, example_instance_to_explain, True, show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_lime_explanations(X_train, gaussian_perturbations[j], model, self.feature_names, self.class_labels, filepath, i, example_instance_to_explain, True, show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)
                elif(explainer == "kernel_shap"):
                    print('Getting Kernel SHAP explanations...')
                    explanations[explainer] = self.get_kernel_shap_explanations(X_train, data_to_explain, predict_function, self.feature_names, filepath, i, example_instance_to_explain)
                    for j, instance in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_kernel_shap_explanations(X_train, perturbations[j], predict_function, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_kernel_shap_explanations(X_train, gaussian_perturbations[j], predict_function, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)
                elif(explainer == "maple"):
                    print('Getting MAPLE explanations...')
                    explanations[explainer] = self.get_maple_explanations(X_train, data_to_explain, predict_function, self.feature_names, filepath, i, example_instance_to_explain)
                    for j, instance in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_maple_explanations(X_train, perturbations[j], predict_function, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_maple_explanations(X_train, gaussian_perturbations[j], predict_function, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)
                elif(explainer == "smoothgrad"):
                    print('Getting Smoothgrad explanations...')
                    explanations[explainer] = self.get_smoothgrad_explanations(data_to_explain, model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain)
                    for j, instaifnce in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_smoothgrad_explanations(perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_smoothgrad_explanations(gaussian_perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)
                elif(explainer == "vanilla_gradients"):
                    print('Getting Vanilla Gradients explanations...')
                    explanations[explainer] = self.get_vanilla_gradients_explanations(data_to_explain, model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain)
                    for j, instance in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_vanilla_gradients_explanations(perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_vanilla_gradients_explanations(gaussian_perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)
                elif(explainer == "guided_backpropagation"):
                    print('Getting Guided Backpropagation explanations...')
                    explanations[explainer] = self.get_guided_backpropagation_explanations(data_to_explain, model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain)
                    for j, instance in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_guided_backpropagation_explanations(perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_guided_backpropagation_explanations(gaussian_perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)
                elif(explainer == "layerwise_relevance_propagation"):
                    print('Getting Layerwise Relevance Propagation explanations...')
                    explanations[explainer] = self.get_layerwise_relevance_propagation_explanations(data_to_explain, model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain)
                    for j, instance in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_layerwise_relevance_propagation_explanations(perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, perturbed=True, show=show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_layerwise_relevance_propagation_explanations(gaussian_perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, perturbed=True, show=show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)
                elif(explainer == "gradients_x_input"):
                    print('Getting Gradients x Input explanations...')
                    explanations[explainer] = self.get_gradients_x_input_explanations(data_to_explain, model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain)
                    for j, instance in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_gradients_x_input_explanations(perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_gradients_x_input_explanations(gaussian_perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)
                elif(explainer == "deep_taylor"):
                    print('Getting Deep Taylor Decomposition explanations...')
                    explanations[explainer] = self.get_deep_taylor_explanations(data_to_explain, model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain)
                    for j, instance in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_deep_taylor_explanations(perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_deep_taylor_explanations(gaussian_perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, True, show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)
                elif(explainer == "integrated_gradients"):
                    print('Getting Integrated Gradients explanations...')
                    explanations[explainer] = self.get_integrated_gradients_explanations(data_to_explain, model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain)
                    for j, instance in enumerate(data_to_explain):
                        show = True

                        if(j > 0):
                            show = False
                        instance_perturbed_explanations = self.get_integrated_gradients_explanations(perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, perturbed=True, show=show)
                        perturbed_explanations[explainer].append(instance_perturbed_explanations)

                        instance_gaussian_perturbed_explanations = self.get_integrated_gradients_explanations(gaussian_perturbations[j], model_without_softmax, self.feature_names, filepath, i, example_instance_to_explain, perturbed=True, show=show)
                        gaussian_perturbed_explanations[explainer].append(instance_gaussian_perturbed_explanations)

            explanations_outer.append(explanations)
            perturbed_explanations_outer.append(perturbed_explanations)
            gaussian_perturbed_explanations_outer.append(gaussian_perturbed_explanations)

            data_to_save = {
                'checkpoint': i,
                'explanations': explanations_outer,
                'perturbed_explanations': perturbed_explanations_outer,
                'gaussian_perturbed_explanations': gaussian_perturbed_explanations_outer,
                'random_samples': random_samples_outer
            }

            self.save_explanation_state(data_to_save, filepath)

            print(f"--------------------------------------------------------------------")
        
        return explanations_outer, perturbed_explanations_outer