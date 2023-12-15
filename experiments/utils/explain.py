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

class Explanation():
    def __init__(self, feature_names, class_labels, explainers_to_use, num_features):
        self.feature_names = feature_names
        self.class_labels = class_labels
        self.explainers_to_use = explainers_to_use
        self.num_features = num_features

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
    
    def display_feature_importance(self, feature_names, feature_importance, filepath, explainer, fold_num, prediction, perturbed=False):
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
            plt.show()
            plt.savefig(f'{filepath}/explanations/figures/perturbed_feature_importance_{explainer}_fold_{fold_num}.png')
        else:
            plt.title(f"Example {explainer} explanation (Instance {prediction}, Fold {fold_num})")
            plt.show()
            plt.savefig(f'{filepath}/explanations/figures/feature_importance_{explainer}_fold_{fold_num}.png')

        return
    
    def display_gradients(self, example, filepath, explainer, fold_num, prediction, perturbed=False):
        perturbed_tag = ""
        perturbed_path = ""
        if(perturbed):
            perturbed_tag = "perturbed"
            perturbed_path = "perturbed_"

        plt.imshow(example, cmap='seismic', interpolation='nearest')
        plt.title(f"Visualisation of {perturbed_tag} gradients using {explainer} (Instance {prediction}, Fold {fold_num})")
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
            plt.show()
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
    


