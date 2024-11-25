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
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.optimizers import Nadam
import shap
import uproot
from plotting.plotter import plotter

# Set TensorFlow and Matplotlib configurations
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
matplotlib.use('Agg')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Seed for reproducibility
np.random.seed(7)

CURRENT_DATETIME = datetime.now()

def GenerateGitPatchAndLog(logFileName,GitPatchName):
    #CMSSWDirPath = os.environ['CMSSW_BASE']
    #CMSSWRel = CMSSWDirPath.split("/")[-1]

    os.system('git diff > '+GitPatchName)

    outScript = open(logFileName,"w");
    #outScript.write('\nCMSSW Version used: '+CMSSWRel+'\n')
    #outScript.write('\nCurrent directory path: '+CMSSWDirPath+'\n')
    outScript.close()

    os.system('echo -e "\n\n============\n== Latest commit summary \n\n" >> '+logFileName )
    os.system("git log -1 --pretty=tformat:' Commit: %h %n Date: %ad %n Relative time: %ar %n Commit Message: %s' >> "+logFileName )
    os.system('echo -e "\n\n============\n" >> '+logFileName )
    os.system('git log -1 --format="SHA: %H" >> '+logFileName )

def load_data_from_EOS(self, directory, mask='', prepend='root://eosuser.cern.ch'):
    eos_dir = '/eos/user/%s ' % (directory)
    eos_cmd = 'eos ' + prepend + ' ls ' + eos_dir
    print(eos_cmd)
    return

# Load data from ROOT files into a DataFrame
def load_data(inputPath,variables, num_events):
    keys = ['sig', 'bckg']
    data = pd.DataFrame(columns=variables)
    for key in keys:
        if key == 'sig':
            fileNames = [
                "GluGluHToZZTo2L2Nu_M1000_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8"
            ]
            target = 1
        else:
            fileNames = [
                "ZZTo2L2Nu",
            ]
            target = 0

        for filen in fileNames:
            if "GluGluHToZZ" in filen:
                process_ID = "ggF"
            elif "VBFHToZZ" in filen:
                process_ID = "VBF"
            elif "ZZTo2L2Nu" in filen:
                process_ID = "ZZ"
            full_path = os.path.join(inputPath, f"{filen}.root")
            if not os.path.exists(full_path):
                print(f"File not found: {full_path}")
                continue
            tree = uproot.open(full_path)["Events"]
            chunk_df = tree.arrays(variables, library="pd", entry_stop=num_events)
            chunk_df['target'] = target
            chunk_df['key']=key
            chunk_df['process_ID']=process_ID
            chunk_df['classweight']=1.0
            data = pd.concat([data, chunk_df], ignore_index=True)
    return data

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

# Custom learning rate scheduler
def custom_learning_rate_scheduler(epoch, lr):
    if epoch < 10:
        return 0.01
    else:
        return float(lr * tf.math.exp(-0.05 * (epoch - 10)))

# Create directory if not exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



# Build and compile a DNN model
def build_model(input_dim, activation='relu', dropout_rate=0.2, learn_rate=0.001):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation=activation),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Nadam(learning_rate=learn_rate), loss='binary_crossentropy', metrics=METRICS)
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

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train a DNN for HH->WWyy analysis.")
    parser.add_argument('--inputPath', required=True, help="Path to input ROOT files.")
    parser.add_argument('--output_dir', required=True, help="Directory to save outputs.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--learn_rate', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--num_events', type=int, default=1000, help="Number of events to load.")
    args = parser.parse_args()

    ensure_directory_exists(args.output_dir)
    variables = ["pTL1", "etaL1", "phiL1"]

    # Load data
    data = load_data(args.inputPath, variables, args.num_events)
    X = data[variables[:-1]].values
    Y = data['target'].values

    # Split into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=7)

    # Build model
    model = build_model(input_dim=X_train.shape[1], learn_rate=args.learn_rate)

    # Train model
    history = train_model(
        model, X_train, Y_train, X_val, Y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir
    )

    # Evaluate and save model
    model.save(os.path.join(args.output_dir, "model.h5"))
    plot_confusion_matrix(Y_val, (model.predict(X_val) > 0.5).astype(int), os.path.join(args.output_dir, "confusion_matrix.png"))

if __name__ == "__main__":
    main()
