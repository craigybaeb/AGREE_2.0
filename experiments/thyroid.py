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

# Define a function to extract the desired substring from the class based on content
def extract_class(row):
    if pd.notna(row):
        if 'negative' in row:
            return 'negative'
        elif 'decreased' in row:
            return 'decreased'
        elif 'increased' in row:
            return 'increased'
    return 'unknown'

def run_thyroid(seeds, num_splits_optimisation, seed, verbose, num_explanations, explainers_to_use):
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
    
    # Multi-class classification
    thyroid = 'experiments/data/allbp.data' 

    thyroid_names = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid', 'sick', 'pregnant', 'surgery', 'treatment', 'hypothyroid', 'hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'tsh_measured', 'tsh', 't3_measured', 't3', 'tt4_measured', 'tt4', 't4u_measured', 't4u', 'fti_measured', 'fti', 'tbg_measured', 'tbg', 'referral', 'class']

    # Load the data from GitHub
    thyroid_df = pd.read_csv(thyroid, names=thyroid_names)

    # Re-name the dataset to plug-and-play with notebook template
    dataset = thyroid_df

    # Extract the actual class from the class column
    dataset['class'] = dataset['class'].apply(extract_class)

    # Get the number of unique values in each column
    unique_counts = dataset.nunique()

    # Find the column names with non-unique values
    non_unique_columns = unique_counts[unique_counts > 1].index

    # Remove columns with non-unique values
    dataset = dataset[non_unique_columns]

    # Replace "?" with NaN values
    dataset.replace("?", np.nan, inplace=True)

    # Check for NaN values in the DataFrame
    nan_check = dataset.isna()

    # Count the number of NaN values in each column
    nan_counts = nan_check.sum()

    # Drop the NaN values from the dataset
    dataset = dataset.dropna()

    # Label encode the categorical values
    label_encoder = LabelEncoder()

    # Include the class, as it needs encoded too
    categories = [
        'sex',
        'on_thyroxine',
        'query_on_thyroxine',
        'on_antithyroid',
        'sick',
        'pregnant',
        'surgery',
        'treatment',
        'hypothyroid',
        'hyperthyroid',
        'lithium',
        'goitre',
        'tumor',
        'hypopituitary',
        'psych',
        'tsh_measured',
        't3_measured',
        'tt4_measured',
        't4u_measured',
        'fti_measured',
        'referral',
        'class'
        ]

    # Do the encoding per column
    for column in categories:
        dataset[column] = label_encoder.fit_transform(dataset[column])

    # Convert the dataset to float
    dataset = dataset.astype(float) 


    # We only want to show the continous features
    continuous_features = [item for item in dataset.columns if item not in categories]

    # Remove the outliers
    dataset = dataset[dataset['age'] <= 100]
    dataset = dataset[dataset['tsh'] <= 300]
    dataset = dataset[dataset['t3'] <= 10]
    dataset = dataset[dataset['tt4'] <= 400]


    # MinMax scale the continuous data to between 0 and 1 to help boost model performance

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the scaler on the column to min-max scale it
    dataset['age'] = scaler.fit_transform(dataset[['age']])
    dataset['tsh'] = scaler.fit_transform(dataset[['tsh']])
    dataset['t3'] = scaler.fit_transform(dataset[['t3']])
    dataset['tt4'] = scaler.fit_transform(dataset[['tt4']])
    dataset['fti'] = scaler.fit_transform(dataset[['fti']])

    # Drop any duplicate rows
    dataset = dataset.drop_duplicates()

    # Split the dataset into data and labels
    x = dataset.drop(["class"], axis=1)
    y = dataset["class"]

    # Correct the class imbalance with SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=seed)
    x_resampled, y_resampled = smote.fit_resample(x, y)

    num_features = 27
    num_classes = 3

    # Set constants
    filepath = 'experiments/results/thyroid'  # Filepath for saving data

    # Get the class labels for the dataset
    class_labels = pd.unique(y_resampled)

    explain = Explanation(thyroid_names, class_labels, explainers_to_use, num_features, filepath)

    state_file = "experiments/results/thyroid/training/results/model_results_and_data.pkl"
    if os.path.exists(state_file):
        with open(state_file, 'rb') as file:
            saved_state = pickle.load(file)

    explanation_state_file = f'{filepath}/explanations/results/explanation_state.pkl'
    checkpoint = 0

    if os.path.exists(explanation_state_file):
        with open(explanation_state_file, 'rb') as file:
            explanation_state = pickle.load(file)
            checkpoint = explanation_state['checkpoint']


    perform_outer_cross_validation(x_resampled, y_resampled, num_classes, num_features, seeds, num_splits_optimisation, filepath)
    explain.get_explanations(x_resampled, y_resampled, seeds, filepath, saved_state['predictions'], checkpoint, num_explanations)



