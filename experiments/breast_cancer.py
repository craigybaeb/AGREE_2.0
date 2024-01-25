# Import the required packages

# Standard library imports
import os
import random
import time
from datetime import datetime, timedelta
from itertools import cycle

# Third-party imports for data handling and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo # To import the dataset
from collections import Counter # Will be used to show the class distribution
from imblearn.over_sampling import SMOTE

# Machine learning and neural network imports
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

# Keras and TensorFlow imports for neural network modeling
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # We need to disable eager execution to work with iNNvestigate later

# Explainers
import shap
import lime
import innvestigate # Smoothgrad, Vanilla Gradients, Input x Gradients, Layerwise Relevance Propagation, Guided Backpropagation, Deep Taylor
from shap import DeepExplainer, GradientExplainer, PermutationExplainer, KernelExplainer

# File storage imports
import pickle
import pathlib
import sys

# Scikit-learn base classes for custom model wrapper
from sklearn.base import BaseEstimator, ClassifierMixin

# Initialising SHAP to surpress warnings
# shap.explainers._deep.deep_tf.op_handlers["SelectV2"] = shap.explainers._deep.deep_tf.passthrough
# shap.explainers._deep.deep_tf.op_handlers["SplitV"] = shap.explainers._deep.deep_tf.passthrough

# Get our custom functions for explaining the models
from utils.train import perform_outer_cross_validation

from utils.explain import Explanation

from utils.robustness import Robustness
robustness = Robustness()

def run_breast_cancer(seeds, num_splits_optimisation, seed, verbose, num_explanations, explainers_to_use):
    # Set constants
    filepath = 'experiments/results/breast_cancer'  # Filepath for saving data

    # Configure the model
    num_classes = 2
    num_features = 30

    # Hyperparameters to test
    param_grid = {
        'architecture': [(64,), (128,), (32, 32), (64, 32), (64, 32, 16), (32, 16, 8), (64, 16, 8), (32, 16, 8, 4), (32, 32, 32), (16, 16)],
        'learning_rate': [0.001, 0.01, 0.0001]
    }

    tf.keras.utils.set_random_seed(seed)

    # Create folders to save results
    pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{filepath}/training/results').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{filepath}/training/figures').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{filepath}/training/models').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{filepath}/explanations/results').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{filepath}/explanations/figures').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{filepath}/results/results').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{filepath}/results/figures').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{filepath}/results/models').mkdir(parents=True, exist_ok=True)

    # Multi-class classification
    import ssl
    # Ignore ssl certificate verification
    ssl._create_default_https_context = ssl._create_unverified_context

    # Fetch dataset
    cancer = fetch_ucirepo(id=17)

    # Data (as pandas dataframes)
    x = cancer.data.features
    y = cancer.data.targets

    # Join the data
    cancer_df = pd.concat([x, y], axis=1)

    # Re-name the dataset to plug-and-play with notebook template
    dataset = cancer_df

    class_mapping = {'M': 1, 'B': 0}

    dataset['Diagnosis'] = dataset['Diagnosis'].map(class_mapping)

    dataset['compactness1'] = np.log(dataset['compactness1'])
    dataset['area1'] = np.log(dataset['area1'])
    dataset['concavity1'] = np.sqrt(dataset['concavity1'])
    dataset['concave_points1'] = np.sqrt(dataset['concave_points1'])
    dataset['fractal_dimension1'] = np.log(dataset['fractal_dimension1'])
    dataset['radius2'] = np.log(dataset['radius2'])
    dataset['texture2'] = np.log(dataset['texture2'])
    dataset['area2'] = np.log(dataset['area2'])
    dataset['perimeter2'] = np.log(dataset['perimeter2'])
    dataset['smoothness2'] = np.log(dataset['smoothness2'])
    dataset['compactness2'] = np.log(dataset['compactness2'])
    dataset['concavity2'] = np.sqrt(dataset['concavity2'])
    dataset['concave_points2'] = np.sqrt(dataset['concave_points2'])
    dataset['symmetry2'] = np.log(dataset['symmetry2'])
    dataset['fractal_dimension2'] = np.log(dataset['fractal_dimension2'])
    dataset['area3'] = np.log(dataset['area3'])
    dataset['compactness3'] = np.log(dataset['compactness3'])
    dataset['concavity3'] = np.sqrt(dataset['concavity3'])
    dataset['fractal_dimension3'] = np.log(dataset['fractal_dimension3'])

    x = dataset.drop(["Diagnosis"], axis=1)
    y = dataset["Diagnosis"]


    smote = SMOTE(sampling_strategy='auto', random_state=seed)
    x_resampled, y_resampled = smote.fit_resample(x, y)

    x_resampled = x_resampled.to_numpy()
    y_resampled = y_resampled.to_numpy()

    # Get the feature names excluding class
    feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

    # Get the class labels for the dataset
    class_labels = pd.unique(y_resampled)

    explain = Explanation(feature_names, class_labels, explainers_to_use, num_features, filepath)

    state_file = "experiments/results/breast_cancer/training/results/model_results_and_data.pkl"
    if os.path.exists(state_file):
        with open(state_file, 'rb') as file:
            saved_state = pickle.load(file)

    splits = saved_state["data_splits"]
    explanation_state_file = f'{filepath}/explanations/results/explanation_state.pkl'
    checkpoint = 0

    if os.path.exists(explanation_state_file):
        with open(explanation_state_file, 'rb') as file:
            explanation_state = pickle.load(file)
            checkpoint = explanation_state['checkpoint']


    perform_outer_cross_validation(x_resampled, y_resampled, num_classes, num_features, seeds, num_splits_optimisation, filepath, param_grid=param_grid)
    explain.get_explanations(x_resampled, y_resampled, seeds, filepath, saved_state['predictions'], checkpoint, num_explanations)