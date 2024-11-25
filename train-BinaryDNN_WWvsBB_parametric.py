import os
import sys
import tempfile
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import json
import argparse

os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Nadam
import uproot
from plotting.plotter import plotter

from plotting.plotter_New import plot_correlation_matrix
from plotting.plotter_New import plot_training_progress
from plotting.plotter_New import plot_metrics
from plotting.plotter_New import plot_confusion_matrix_multiclass
from plotting.plotter_New import plot_roc_curve_multiclass
from plotting.plotter_New import plot_shap_values
from plotting.plotter_New import plot_overfitting
from plotting.plotter_New import plot_overfitting_per_class
from plotting.plotter_New import plot_overfitting_multiclass
from plotting.plotter_New import plot_classifier_output

# Set TensorFlow and Matplotlib configurations
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
matplotlib.use('Agg')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Seed for reproducibility
np.random.seed(7)

CURRENT_DATETIME = datetime.now()

class PerMassMetricsCallback(Callback):
    def __init__(self, model_reference, X_val, Y_val, masses, output_dir):
        """
        Callback to compute accuracy and loss for each mass point during training.

        :param model_reference: Reference to the model being trained.
        :param X_val: Validation data features (including mass as a feature).
        :param Y_val: Validation data labels (one-hot encoded).
        :param masses: List of unique mass points.
        :param output_dir: Directory to save plots.
        """
        super().__init__()
        self.model_reference = model_reference  # Store the model using a different attribute name
        self.X_val = X_val
        self.Y_val = Y_val
        self.masses = masses
        self.output_dir = output_dir
        self.history = {mass: {'accuracy': [], 'loss': []} for mass in masses}

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEvaluating per mass metrics after epoch {epoch + 1}...")
        for mass in self.masses:
            # Filter validation data by mass
            mass_filter = self.X_val[:, -1] == mass
            X_mass = self.X_val[mass_filter]
            Y_mass = self.Y_val[mass_filter]

            if len(X_mass) == 0:
                continue  # Skip if no data for this mass

            # Compute loss and accuracy for this mass
            metrics = self.model.evaluate(X_mass, Y_mass, verbose=0)
            self.history[mass]['loss'].append(metrics[0])
            self.history[mass]['accuracy'].append(metrics[1])

    def plot_metrics(self):
        """
        Generate and save accuracy vs. epoch and loss vs. epoch plots for each mass.
        """
        for mass in self.masses:
            if len(self.history[mass]['accuracy']) == 0:
                continue  # Skip if no data for this mass

            # Plot accuracy
            plt.figure(figsize=(8, 6))
            plt.plot(self.history[mass]['accuracy'], label=f"Mass {mass}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy vs. Epoch for Mass {mass}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/accuracy_vs_epoch_mass_{mass}.png")
            plt.close()

            # Plot loss
            plt.figure(figsize=(8, 6))
            plt.plot(self.history[mass]['loss'], label=f"Mass {mass}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss vs. Epoch for Mass {mass}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/loss_vs_epoch_mass_{mass}.png")
            plt.close()

            print(f"Saved accuracy and loss plots for mass {mass}")


# Ensure directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load data from ROOT files into a DataFrame
def load_data(inputPath, variables, num_events, csv_path, metadata_path, signal_masses):
    """
    Load data for ggH, VBF, and Background processes. Background mass is randomly sampled from the signal mass range.

    :param inputPath: Path to the ROOT files.
    :param variables: List of variables to extract from the ROOT files.
    :param num_events: Number of events to read.
    :param csv_path: Path to save/load the CSV file.
    :param metadata_path: Path to save/load the metadata (variable list).
    :param signal_masses: List of signal masses for ggH and VBF.
    :return: Pandas DataFrame containing the dataset.
    """
    csv_exists = os.path.exists(csv_path)
    metadata_exists = os.path.exists(metadata_path)

    # Check if the CSV file exists and the variables are unchanged
    if csv_exists and metadata_exists:
        with open(metadata_path, 'r') as metadata_file:
            saved_metadata = json.load(metadata_file)

        saved_variables = saved_metadata.get('variables', [])
        if set(variables) == set(saved_variables):
            print(f"Loading data from existing CSV: {csv_path}")
            return pd.read_csv(csv_path)
        else:
            print("Variable list has changed. Reloading data from ROOT files.")
    else:
        print("CSV or metadata not found. Reading data from ROOT files.")

    # Read data from ROOT files
    keys = ['ggh', 'vbf', 'bkg']
    data = pd.DataFrame(columns=variables + ['target', 'process_ID', 'classweight', 'mass'])
    for key in keys:
        if key == 'ggh':
            fileNames = [f"GluGluHToZZTo2L2Nu_M{mass}_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8" for mass in signal_masses]
            target = 0  # ggH
        elif key == 'vbf':
            fileNames = [f"VBF_HToZZTo2L2Nu_M{mass}_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8" for mass in signal_masses]
            target = 1  # VBF
        else:  # Background
            fileNames = ["ZZTo2L2Nu"]
            target = 2  # Background

        for filen in fileNames:
            process_ID = key.upper()
            full_path = os.path.join(inputPath, f"{filen}.root")
            if not os.path.exists(full_path):
                print(f"File not found: {full_path}")
                continue
            tree = uproot.open(full_path)["Events"]
            chunk_df = tree.arrays(variables, library="pd", entry_stop=num_events)
            chunk_df['target'] = target
            chunk_df['process_ID'] = process_ID
            chunk_df['classweight'] = 1.0

            mass = None
            # Assign mass for signals
            if key != 'bkg':
                mass = int(filen.split("_M")[1].split("_")[0])  # Extract mass from filename for signals
                chunk_df['mass'] = mass
            else:  # Assign mass for background randomly sampled from signal masses
                chunk_df['mass'] = np.random.choice(signal_masses, size=len(chunk_df))

            data = pd.concat([data, chunk_df], ignore_index=True)

            print(f"Loaded {len(chunk_df)} events for process {key} with mass {mass}")

    # Save DataFrame to CSV
    print(f"Saving DataFrame to CSV: {csv_path}")
    data.to_csv(csv_path, index=False)

    # Save variable list to metadata
    metadata = {"variables": variables}
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    return data

# Ensure input data is numeric and clean
def preprocess_data(data):
    # Replace NaN or infinite values with a default (e.g., 0 or mean)
    data = data.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
    data = data.fillna(0)  # Replace NaN with 0 (or use column mean if needed)
    return data.astype('float32')  # Ensure all values are float32


# Metrics for evaluation
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.AUC(name='auc'),
]

# Custom learning rate scheduler
def custom_learning_rate_scheduler(epoch, lr):
    if epoch < 10:
        return 0.01
    else:
        return float(lr * tf.math.exp(-0.05 * (epoch - 10)))

# Build and compile a multi-class DNN model
def build_model(input_dim, activation='relu', dropout_rate=0.2, learn_rate=0.001):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation=activation),
        Dense(3, activation="softmax")  # Softmax for multi-class classification
    ])
    model.compile(optimizer=Nadam(learning_rate=learn_rate), loss='categorical_crossentropy', metrics=METRICS)
    return model

def build_parametric_model(input_dim, activation='relu', dropout_rate=0.2, learn_rate=0.001):
    """
    Build a parametric multi-class DNN model with an additional mass input.
    """
    model = Sequential([
        Dense(256, input_dim=input_dim, activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation=activation),
        Dense(3, activation="softmax")  # Softmax for multi-class classification
    ])
    model.compile(optimizer=Nadam(learning_rate=learn_rate), loss='categorical_crossentropy', metrics=METRICS)
    return model

# Train the model with early stopping
def train_model(model, X_train, Y_train, X_val, Y_val, batch_size, epochs, output_dir, class_weight=None):
    early_stopping = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
    csv_logger = CSVLogger(os.path.join(output_dir, 'training.log'))
    lr_scheduler = LearningRateScheduler(custom_learning_rate_scheduler)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, csv_logger, lr_scheduler],
        class_weight=class_weight,
        verbose=1
    )
    return history

def train_model_with_mass_metrics(model, X_train, Y_train, X_val, Y_val, batch_size, epochs, output_dir, class_weight=None):
    """
    Train the model with accuracy and loss tracking for each mass point.

    :param model: The model to train.
    :param X_train: Training feature set (including the mass feature).
    :param Y_train: Training labels (one-hot encoded).
    :param X_val: Validation feature set (including the mass feature).
    :param Y_val: Validation labels (one-hot encoded).
    :param batch_size: Batch size for training.
    :param epochs: Number of epochs to train.
    :param output_dir: Directory to save plots and metrics.
    :param class_weight: Optional class weights for imbalanced data.
    :return: Training history and per mass metrics callback.
    """
    masses = np.unique(X_val[:, -1])
    per_mass_callback = PerMassMetricsCallback(model, X_val, Y_val, masses, output_dir)

    # Train the model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[per_mass_callback],
        class_weight=class_weight,
        verbose=1
    )

    # Generate plots for per mass metrics
    per_mass_callback.plot_metrics()

    return history, per_mass_callback

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, output_path, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train a multi-class DNN for ggH/VBF/background classification.")
    parser.add_argument('--inputPath', required=True, help="Path to input ROOT files.")
    parser.add_argument('--output_dir', required=True, help="Directory to save outputs.")
    parser.add_argument('--job_name', type=str, default="DNN", help="Job name.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--learn_rate', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--num_events', type=int, default=1000, help="Number of events to load.")
    parser.add_argument('--json', type=str, default='input_variables.json', help="Input variable JSON file.")

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, f"{args.job_name}")
    ensure_directory_exists(args.output_dir)

    # Create list of headers for dataset .csv
    input_var_jsonFile = open(args.json,'r')
    variable_list = json.load(input_var_jsonFile).items()
    variables = []
    for key,var in variable_list:
        variables.append(key)

    print(f"Variables: {variables}")

    # Define paths and parameters
    csv_path = os.path.join(args.output_dir, "output_dataframe.csv")
    metadata_path = os.path.join(args.output_dir, "variables_metadata.json")
    model_path = os.path.join(args.output_dir, "model.h5")

    # Load data
    signal_masses = [300, 400, 500, 1500, 2000, 3000]
    data = load_data(args.inputPath, variables, args.num_events, csv_path=csv_path, metadata_path=metadata_path, signal_masses=signal_masses)

    # print dataframe info
    print(data.info())

    # print dataframe head
    print(data.head())
    print(data['mass'].unique())

    # Define feature columns explicitly (these are the features used for training)
    feature_columns = [col for col in variables if col not in ['target', 'process_ID', 'classweight', 'mass']]

    # add mass to the feature columns
    feature_columns.append('mass')

    print(f"Feature columns: {feature_columns}")

    # Extract only the features used for training
    X = data[feature_columns].values
    Y = pd.get_dummies(data['target']).values  # One-hot encoding for multi-class

    # Split into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=7)

    # Debugging: Ensure 'mass' is in X
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("Mass in validation data (last column of X_val):", X_val[:, -1])

    # Preprocess training and validation data
    X_train = preprocess_data(pd.DataFrame(X_train)).values
    X_val = preprocess_data(pd.DataFrame(X_val)).values
    Y_train = preprocess_data(pd.DataFrame(Y_train)).values
    Y_val = preprocess_data(pd.DataFrame(Y_val)).values

    # Create plots directory
    plots_dir = os.path.join(args.output_dir, "plots")
    ensure_directory_exists(plots_dir)

    # Check if the model already exists
    if os.path.exists(model_path):
        print(f"Trained model already exists at {model_path}. Loading the model...")
        model = load_model(model_path)
    else:
        # Build model
        model = build_model(input_dim=X_train.shape[1], learn_rate=args.learn_rate)

        # # Train model
        # history = train_model(
        #     model, X_train, Y_train, X_val, Y_val,
        #     batch_size=args.batch_size,
        #     epochs=args.epochs,
        #     output_dir=args.output_dir
        # )

        # Train model with per mass metrics
        history, per_mass_callback = train_model_with_mass_metrics(
            model, X_train, Y_train, X_val, Y_val,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=plots_dir
        )

        # Evaluate and save model
        model.save(model_path)
        print(f"Saved model to: {model_path}")

    for mass in signal_masses:
        print(f"Generating validation plfots for mass: {mass}")

        # Filter data by mass
        mass_filter = X_val[:, -1] == mass  # The last column is the 'mass'
        X_val_mass = X_val[mass_filter]


        print(f"len(feature_columns) = {len(feature_columns)}")
        # Extract only the features used for training
        X_val_mass_features = X_val_mass[:, :len(feature_columns)]  # Use only the first N columns corresponding to feature_columns
        Y_val_mass = Y_val[mass_filter]

        if len(X_val_mass) == 0:
            print(f"No validation data found for mass: {mass}")
            continue

        # print feature_columns
        print(f"Feature columns: {feature_columns}")

        # 	2.	Validate X_train.shape and X_val_mass_features.shape to ensure they match input_dim=14.
        print("X_train shape:", X_train.shape)
        print("X_val_mass_features shape:", X_val_mass_features.shape)

        # Evaluate model
        y_pred_mass = np.argmax(model.predict(X_val_mass_features), axis=1)  # Exclude 'mass' from prediction input
        y_true_mass = np.argmax(Y_val_mass, axis=1)
        y_score_mass = model.predict(X_val_mass_features)

        # ROC Curve
        plot_roc_curve_multiclass(Y_val_mass, y_score_mass, plots_dir, labels=["ggH", "VBF", "Background"], mass=mass)

        # Confusion Matrix
        plot_confusion_matrix_multiclass(Y_val_mass, y_pred_mass, plots_dir, labels=["ggH", "VBF", "Background"], mass=mass)

        # Classification Report
        plot_classifier_output(model, X_train, Y_train, X_val_mass_features, Y_val_mass, output_dir=plots_dir, mass=mass)

if __name__ == "__main__":
    main()
