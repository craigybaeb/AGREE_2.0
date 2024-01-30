# Import the required packages

# Explainers
import pathlib
import shap
import lime
import innvestigate # Smoothgrad, Vanilla Gradients, Input x Gradients, Layerwise Relevance Propagation, Guided Backpropagation, Deep Taylor
from shap import DeepExplainer, GradientExplainer, PermutationExplainer, KernelExplainer

# For data handling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE # Used for correcting class imbalance
from collections import Counter # Will be used to show the class distribution
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo # To import the dataset

# For training the Multi Layer Perceptron model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping

from utils.explain import Explanation
from utils.train import perform_outer_cross_validation
tf.compat.v1.disable_eager_execution() # We need to disable eager execution to work with iNNvestigate later

# For fitting a k-NN in the evaluation
from sklearn.neighbors import KNeighborsClassifier

# For evaluating the model
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score,
    roc_auc_score
)

# For saving/fetching files from Google Drive
import pickle

# For visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# To set the seed
import random

# Initialising SHAP to surpress warnings
shap.explainers._deep.deep_tf.op_handlers["SelectV2"] = shap.explainers._deep.deep_tf.passthrough
shap.explainers._deep.deep_tf.op_handlers["SplitV"] = shap.explainers._deep.deep_tf.passthrough

import os

def run_mushroom(seeds, num_splits_optimisation, seed, verbose, num_explanations, explainers_to_use):
    # Hyperparameters to test
    param_grid = {
        'architecture': [(64,), (128,), (32, 32), (64, 32), (64, 32, 16), (32, 16, 8), (64, 16, 8), (32, 16, 8, 4), (32, 32, 32), (16, 16)],
        'learning_rate': [0.001, 0.01, 0.0001]
    }

    # Set constants
    filepath = 'experiments/results/mushroom'  # Filepath for saving data
    
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

    # Binary classification
    mushroom = '/content/mushroom.csv'

    # Load the data from GitHub
    mushroom_df = pd.read_csv(mushroom, sep=";")

    # Re-name the dataset to plug-and-play with notebook template
    dataset = mushroom_df

    # Get the number of unique values in each column
    unique_counts = dataset.nunique()

    # Check for NaN values in the DataFrame
    nan_check = dataset.isna()

    # Count the number of NaN values in each column
    nan_counts = nan_check.sum()

    # Drop columns with too many NaNs
    dataset = dataset.drop(["spore-print-color", "veil-color", "cap-surface", "stem-surface", "stem-root", "gill-spacing"], axis=1)

    # Remove the remainder rows containing NaNs
    dataset = dataset.dropna()

    # Label encode the categorical values
    label_encoder = LabelEncoder()

    # Define the categorical features
    categories = [
    'cap-shape',
    'cap-color',
    'does-bruise-or-bleed',
    'gill-attachment',
    'gill-color',
    'stem-color',
    'has-ring',
    'ring-type',
    'habitat',
    'season',
    'class'
    ]

    # Do the encoding per column
    for column in categories:
        dataset[column] = label_encoder.fit_transform(dataset[column])

    # Min max scale continuous features

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # We only want to show the continous features
    continuous_features = [item for item in dataset.columns if item not in categories]

    # Scale the continuous features
    dataset[continuous_features] = scaler.fit_transform(dataset[continuous_features])

    # Drop any duplicate rows
    dataset = dataset.drop_duplicates()
    dataset.info() # Check the state of the data now

    # Split the dataset into data and labels
    x = dataset.drop(["class"], axis=1)
    y = dataset["class"]

    # Check the class distribution

    # Count the number of instances in each class
    class_counts = dataset['class'].value_counts()

    # Correct the class imbalance with SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)

    x_resampled = x_resampled.to_numpy()
    y_resampled = y_resampled.to_numpy()
    
    # Checking the class distribution has been corrected

    # Counting the number of instances in each class
    class_counts = y_resampled.value_counts()

    # Get the feature names excluding class
    feature_names = [col for col in dataset.columns if col != 'cid']

    # Get the class labels for the dataset
    class_labels = pd.unique(y_resampled)

    num_features = 13
    num_classes = 2

    explain = Explanation(feature_names, class_labels, explainers_to_use, num_features, filepath)

    state_file = "experiments/results/aids/training/results/model_results_and_data.pkl"
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


    perform_outer_cross_validation(x_resampled, y_resampled, num_classes, num_features, seeds, num_splits_optimisation, filepath)
    explain.get_explanations(x_resampled, y_resampled, seeds, filepath, saved_state['predictions'], checkpoint, num_explanations)


