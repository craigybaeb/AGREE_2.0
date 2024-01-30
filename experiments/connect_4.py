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

def getNames():
  X = ['a','b','c', 'd', 'e', 'f', 'g']
  names = []

  for x in X:
    for i in range(6):
      NAME = x + str(i)
      names.append(NAME)
  names.append('class')
  return names

def run_connect_4(seeds, num_splits_optimisation, seed, verbose, num_explanations, explainers_to_use):
    # Hyperparameters to test
    param_grid = {
        'architecture': [(64,), (128,), (32, 32), (64, 32), (64, 32, 16), (32, 16, 8), (64, 16, 8), (32, 16, 8, 4), (32, 32, 32), (16, 16)],
        'learning_rate': [0.001, 0.01, 0.0001]
    }
    
    tf.keras.utils.set_random_seed(seed)

    # Set constants
    filepath = 'experiments/results/connect_4'  # Filepath for saving data

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

    #Multi-class classification
    connect4 = '/content/drive/My Drive/PhD/Disagreement Problem/eval/connect4/connect-4.data'
    connect4_df = pd.read_csv(connect4, names=getNames())

    dataset = connect4_df
    dataset = dataset.apply(LabelEncoder().fit_transform)

    x = dataset.drop("class", axis=1)
    x = pd.get_dummies(x)
    y = dataset["class"]

    # Assuming you have a DataFrame named df with a 'class_column'
    class_counts = dataset['class'].value_counts()

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)

    x_resampled = x_resampled.to_numpy()
    y_resampled = y_resampled.to_numpy()
    
    num_classes = 3
    num_features = 42

    # Get the feature names excluding class
    feature_names = [col for col in dataset.columns if col != 'class']

    # Get the class labels for the dataset
    class_labels = pd.unique(y_resampled)

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


