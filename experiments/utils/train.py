# Standard library imports
import os
import time
from datetime import datetime, timedelta

# Third-party imports for data handling and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning and neural network imports
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    f1_score, roc_auc_score, recall_score, precision_score,
    accuracy_score, confusion_matrix, roc_curve
)

# Keras and TensorFlow imports for neural network modeling
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # We need to disable eager execution to work with iNNvestigate later

# Import for model serialization
import pickle

# Scikit-learn base classes for custom model wrapper
from sklearn.base import BaseEstimator, ClassifierMixin

def create_model(num_classes, num_features, architecture=(100,), learning_rate=0.001):
    model = Sequential()
    model.add(Dense(architecture[0], input_shape=(num_features,), activation='relu'))
    for nodes in architecture[1:]:
        model.add(Dense(nodes, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def save_data(data_to_save, filepath):

  for i, model in enumerate(data_to_save['models']):
    model.model.save(f'{filepath}/training/models/model_{i}.h5')

  data_to_save.pop('models', None)

  with open(f'{filepath}/training/results/model_results_and_data.pkl', 'wb') as file:
      pickle.dump(data_to_save, file)
  print(f"Data saved to '{filepath}/training/results/model_results_and_data.pkl'")

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes, num_features, architecture=(100,), learning_rate=0.001, verbose=0):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_features = num_features
        self.verbose = verbose
        self.model = None

    def fit(self, X, y, **kwargs):
        self.history_callback = HistoryCallback()
        self.model = create_model(num_classes=self.num_classes, num_features=self.num_features, architecture=self.architecture, learning_rate=self.learning_rate)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X, y, epochs=100, batch_size=20, verbose=self.verbose, callbacks=[early_stopping, self.history_callback], validation_split=0.1, **kwargs)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def score(self, X, y):
        # Convert y to categorical inside the method
        _, accuracy = self.model.evaluate(X, y, verbose=0)
        return accuracy

class HistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_history = {}

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_history[epoch] = logs

# Function to convert param dict to a string
def param_to_string(param):
    return ', '.join([f"{key}: {value}" for key, value in param.items()])

# Function to load saved state if it exists
def load_saved_state(filepath, resume, outer_loops):
    state_file = f'{filepath}/model_results_and_data.pkl'
    
    if resume:
      if os.path.exists(state_file):
        with open(state_file, 'rb') as file:
            saved_state = pickle.load(file)

            if(saved_state['outer_fold'] < outer_loops):
              print("Resuming from the last saved state.")
            else:
              raise Exception("Can't resume training, number of outer loops already exceeded.")
              
        return saved_state
      else:
        print("Cannot find file to load. Starting from the beginning.")
        return None
    else:
        print("Starting from the beginning.")
        return None
    
# Function to plot training and validation loss and accuracy
def plot_training_validation_metrics(training_loss, validation_loss, training_accuracy, validation_accuracy, fold_num, filepath):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy for Outer Fold {fold_num}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title(f'Training and Validation Loss for Outer Fold {fold_num}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'{filepath}/training/figures/training_validation_fold_{fold_num}.png')
    plt.show()

# Function to plot grid search scores
def plot_grid_search_scores(param, distinct_values, data, filepath):
    plt.figure(figsize=(10, 6))
    sorted_values = sorted(list(distinct_values))
    param_indices = {v: i for i, v in enumerate(sorted_values)}
    x_indices = [param_indices[val] for val in data['params']]

    plt.errorbar(x=x_indices, y=data['means'], yerr=data['stds'], fmt='o', label='Mean Score')

    plt.xticks(range(len(sorted_values)), sorted_values, rotation=45)
    plt.xlabel(f'Value of {param}')
    plt.ylabel('Mean Score')
    plt.title(f'Grid Search Scores by {param}')
    plt.legend()
    plt.savefig(f'{filepath}/training/figures/grid_search_scores_{param}.png')
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(cm, fold_num, filepath):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f'Confusion Matrix for Outer Fold {fold_num}')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(f'{filepath}/training/figures/confusion_matrix_fold_{fold_num}.png')
    plt.show()

# Function to log time-related information
def log_time_info(fold_start_time, start_time, outer_fold):
    fold_elapsed = time.time() - fold_start_time
    total_elapsed = time.time() - start_time
    average_time_per_fold = total_elapsed / outer_fold
    estimated_remaining_time = average_time_per_fold * (5 - outer_fold)
    expected_end_time = datetime.now() + timedelta(seconds=estimated_remaining_time)

    print(f"Fold {outer_fold} completed in {fold_elapsed:.2f} seconds")
    print(f"Total elapsed time: {total_elapsed:.2f} seconds")
    print(f"Estimated remaining time: {estimated_remaining_time:.2f} seconds")
    print(f"Expected end time: {expected_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def plot_metrics_variance(accuracy_scores, f1_scores, recall_scores, auc_scores, precision_scores, filepath):
    # Organize the data into a DataFrame
    data = {
        'Accuracy': accuracy_scores,
        'F1 Score': f1_scores,
        'Recall': recall_scores,
        'AUC': auc_scores,
        'Precision': precision_scores
    }

    df = pd.DataFrame(data)

    # Melting the DataFrame to make it suitable for sns.boxplot
    df_melted = df.melt(var_name='Metric', value_name='Score')

    # Plotting using Seaborn
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Metric', y='Score', data=df_melted)

    # Adding labels and title
    plt.title('Variance in Performance Metrics Between Best Estimators')
    plt.grid(True)
    plt.savefig(f'{filepath}/training/figures/metrics_variance_outer_loop.png')  # Save figure
    plt.show()

def perform_grid_search(X, y, num_splits, num_classes, num_features, param_grid, filepath, seed=42, verbose=3):
    # Cross-Validation (Hyperparameter Tuning)
    cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    model = KerasClassifierWrapper(num_classes, num_features)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, verbose=verbose)

    # Fit the model
    grid.fit(X, y)

    # Get the best model
    best_params = grid.best_params_
    best_model = grid.best_estimator_

    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    learning_rates = [param['learning_rate'] for param in params]  # Extract learning rates

    all_scores = []

    # Organize scores by parameter
    scores_by_param = {}
    distinct_values = {}

    for mean, std, param, lr in zip(means, stds, params, learning_rates):
        all_scores.append((mean, std, param))
        for key, value in param.items():
            if key not in scores_by_param:
                scores_by_param[key] = {'params': [], 'means': [], 'stds': [], 'learning_rate': []}
                distinct_values[key] = set()
            scores_by_param[key]['params'].append(value)
            scores_by_param[key]['means'].append(mean)
            scores_by_param[key]['stds'].append(std)
            scores_by_param[key]['learning_rate'].append(lr)
            distinct_values[key].add(value)

    # Plot grid search scores for each parameter
    for param, data in scores_by_param.items():
        plot_grid_search_scores(param, distinct_values[param], data, filepath)

    grid_search_data = {
        "best_params": best_params,
        "all_scores": all_scores
    }

    with open(f'{filepath}/training/grid_search_results.pkl', 'wb') as file:
        pickle.dump(grid_search_data, file)

    return best_model, best_params


def perform_outer_cross_validation(X, y, num_classes, num_features, num_splits_outer, num_splits_inner, filepath, resume=True, verbose=1, seed=42, param_grid ={
    'architecture': [(64,), (128,), (32, 32), (64, 32), (64, 32, 16), (32, 16, 8), (64, 16, 8), (32, 16, 8, 4), (32, 32, 32), (16, 16)],
    'learning_rate': [0.001, 0.01, 0.0001]
}):
    # Define the number of outer folds
    outer_cv = StratifiedKFold(n_splits=num_splits_outer, shuffle=True, random_state=seed)

    # Initialize variables to store metrics and results
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []
    train_scores_outer = []
    models = []
    matrices = []
    model_histories = []
    splits = []
    preds_outer = []
    outer_fold = 1

    # Load saved state if resuming
    saved_state = load_saved_state(filepath, resume, num_splits_outer)
    if saved_state:
        outer_fold = saved_state["outer_fold"] + 1
        accuracy_scores = saved_state["accuracy_scores"]
        f1_scores = saved_state["f1_scores"]
        precision_scores = saved_state["precision_scores"]
        recall_scores = saved_state["recall_scores"]
        auc_scores = saved_state["auc_scores"]
        models = saved_state["models"]
        matrices = saved_state["matrices"]
        model_histories = saved_state["model_histories"]
        splits = saved_state["data_splits"]
        preds_outer = saved_state["predictions"]
        train_scores_outer = saved_state["train_scores"]


    # Perform the grid search
    _, best_params = perform_grid_search(X, y, num_splits_inner, num_classes, num_features, param_grid, filepath)

    # Record the start time
    start_time = time.time()

    # Loop through each outer fold
    for train_index, test_index in (splits if splits and resume == True else outer_cv.split(X, y)):
        # Check if resuming and some iterations have already been completed
        if outer_fold < len(splits) and resume == True:
            print(f"Skipping Outer Fold {outer_fold} (already completed)")
            outer_fold += 1
            continue

        splits.append((train_index, test_index))

        # Record the start time of the current fold
        fold_start_time = time.time()
        print(f"Outer Fold {outer_fold}")

        # Split the data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fitting with the best parameters from the grid search
        model = KerasClassifierWrapper(num_classes, num_features, **best_params)

        # Fit the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_test_cat = to_categorical(y_test, num_classes=3)

        preds_outer.append(y_pred)

        # Calculate additional metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test_cat, model.model.predict(X_test), multi_class='ovr')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)

        # Store the metrics
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        f1_scores.append(f1)
        recall_scores.append(recall)
        auc_scores.append(auc)

        print(f"Fold {outer_fold} - Best Model Performance:")
        print(f"Accuracy: {accuracy}, F1 Score: {f1}, AUC: {auc}, Recall: {recall}, Precision: {precision}\n")

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        matrices.append(cm)

        # Plot the confusion matrix
        plot_confusion_matrix(cm, outer_fold, filepath)

        # Retrieve the training history from the best model
        model_hist = model.history_callback.epoch_history

        models.append(model)
        model_histories.append(model_hist)

        # Plot training and validation loss and accuracy
        plt.figure(figsize=(12, 5))

        # Plot the training and validation metrics
        plot_training_validation_metrics(
            [epoch_data['loss'] for epoch_data in model_hist.values()],
            [epoch_data['val_loss'] for epoch_data in model_hist.values()],
            [epoch_data['accuracy'] for epoch_data in model_hist.values()],
            [epoch_data['val_accuracy'] for epoch_data in model_hist.values()],
            outer_fold,
            filepath
        )

        # Create a dictionary of all items to be saved
        data_to_save = {
            "accuracy_scores": accuracy_scores,
            "f1_scores": f1_scores,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "auc_scores": auc_scores,
            "models": models,
            "model_histories": model_histories,
            "data_splits": splits,
            "train_scores": train_scores_outer,
            "predictions": preds_outer,
            "matrices": matrices,
            "outer_fold": outer_fold
        }

        # Save data to a file
        save_data(data_to_save, filepath)

        # Log time-related information
        log_time_info(fold_start_time, start_time, outer_fold)

        # Increment the outer fold counter
        outer_fold += 1
    
    plot_metrics_variance(accuracy_scores, f1_scores, recall_scores, auc_scores, precision_scores, filepath)