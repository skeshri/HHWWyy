#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### Importing all the required libraries

import numpy as np
import uproot
import os
import re
import random
from glob import glob
import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import ConcatDataset

import json


# In[2]:


# Function to load the JSON file
def load_json_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


# In[3]:


def _extract_json(data_dir,filename):
    sample_dic = load_json_file(filename)
    samples = []
    print(data_dir)
    for key, value in sample_dic.items():
        sample = os.path.join(data_dir,key)
        #print(sample)
        samples.append([sample,value[0],value[1]])
    #print(samples)
    return samples


# In[4]:


class MultiROOTDatasetWithLabels(Dataset):
    def __init__(self, file_paths, tree_name, features, chunk_size, signal_mass_points=None, label=None, transform=None):
        """
        Args:
            file_paths (list): List of ROOT file paths.
            tree_name (str): Name of the tree to load data from.
            chunk_size (int): Number of entries to load in each chunk.
            signal_mass_points (list, optional): List of signal mass points (for assigning to background).
            label (int): Label for the dataset (e.g., 0 for background, 1 for signal).
            transform (callable, optional): Transform to apply to each sample.
        """
        self.file_paths = file_paths
        self.tree_name = tree_name
        self.chunk_size = chunk_size
        self.label = label
        self.signal_mass_points = signal_mass_points
        self.transform = transform
        self.features = features

        # Extract mass values from filenames if signal dataset
        self.mass_values = []
        self.file_handles = []
        self.file_event_counts = []

        if signal_mass_points is None and label == 1:  # For signal dataset
            self.signal_mass_points = []
            mass_pattern = r"_M(\d+)_"
            for file_path in file_paths:
                match = re.search(mass_pattern, file_path)
                if match:
                    mass_value = float(match.group(1))
                    self.signal_mass_points.append(mass_value)
                else:
                    raise ValueError(f"Mass value not found in filename: {file_path}")
            self.mass_values = self.signal_mass_points

        for file_path in file_paths:
            # Open ROOT file and get event count
            with uproot.open(file_path) as file:
                tree = file[tree_name]
                self.file_event_counts.append(tree.num_entries)
                self.file_handles.append(tree)

        # Calculate cumulative event ranges for indexing
        self.total_events = sum(self.file_event_counts)
        self.cumulative_event_counts = np.cumsum(self.file_event_counts)

        # Initialize for chunk loading
        self.current_chunk = None
        self.current_file_idx = -1
        self.chunk_start_idx = 0

    def __len__(self):
        return self.total_events

    def _get_file_and_local_idx(self, global_idx):
        # Determine which file the global index corresponds to
        file_idx = np.searchsorted(self.cumulative_event_counts, global_idx, side="right")
        if file_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cumulative_event_counts[file_idx - 1]
        return file_idx, local_idx

    def __getitem__(self, idx):
        # Get file index and local index within the file
        file_idx, local_idx = self._get_file_and_local_idx(idx)

        # Load a new chunk if needed
        if self.current_chunk is None or file_idx != self.current_file_idx or not (self.chunk_start_idx <= local_idx < self.chunk_start_idx + self.chunk_size):
            self.current_file_idx = file_idx
            self.chunk_start_idx = (local_idx // self.chunk_size) * self.chunk_size
            start = self.chunk_start_idx
            stop = min(start + self.chunk_size, self.file_event_counts[file_idx])

            # Load the chunk
            self.current_chunk = self.file_handles[file_idx].arrays(self.features, entry_start=start, entry_stop=stop, library="np")

        # Fetch the sample within the chunk
        local_chunk_idx = local_idx - self.chunk_start_idx
        sample = {key: self.current_chunk[key][local_chunk_idx] for key in self.current_chunk}

        # Assign mass value
        if self.label == 1:  # Signal dataset
            sample["mass"] = self.mass_values[file_idx]
        elif self.label == 0:  # Background dataset
            sample["mass"] = random.choice(self.signal_mass_points)

        # Add the label
        sample["label"] = self.label

        # Apply transformations if needed
        if self.transform:
            sample = self.transform(sample)

        return sample


# In[5]:


data_dir = "/eos/user/s/skeshri/SWAN_projects/XZZ2l2nuAnalysis/Data/"
path = "/eos/user/a/avijay/HZZ_mergedrootfiles/"
tree_name = "Events"


# In[6]:


samples = _extract_json(path,"/eos/user/s/skeshri/SWAN_projects/XZZ2l2nuAnalysis/Data/data_full.json")


# In[7]:


variables = load_json_file(data_dir+"/"+"input_variables_test.json")
variables = list(variables.keys())


# In[8]:


backgrounds = [sample[0] for sample in samples if sample[1]=='background']
signals = [sample[0] for sample in samples if sample[1]=='signal']


# In[9]:


tree_name = "Events"
chunk_size = 1000
signal_dataset = MultiROOTDatasetWithLabels(signals, tree_name,variables, chunk_size, label=1)
signal_mass_points=signal_dataset.signal_mass_points
background_dataset = MultiROOTDatasetWithLabels(backgrounds, tree_name, variables,chunk_size,signal_mass_points,label=0)


# In[10]:


import matplotlib.pyplot as plt

data = [background_dataset[i]['mass'] for i in range(500)]
plt.hist(data)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


# In[11]:


combined_dataset = ConcatDataset([background_dataset, signal_dataset])
print(combined_dataset[-1])


# In[12]:


from torch.utils.data import DataLoader, WeightedRandomSampler

# Define weights for balanced sampling
weights = [1.0 / len(background_dataset)] * len(background_dataset) + [1.0 / len(signal_dataset)] * len(signal_dataset)
print(len(weights))
sampler = WeightedRandomSampler(weights, num_samples=len(combined_dataset), replacement=True)
print(len(sampler))
# DataLoader
balanced_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

print(len(balanced_loader))
# Iterate over the DataLoader

# In[13]:


class ParametrizedDNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=[64, 128, 64], output_dim=2):
        super(ParametrizedDNN, self).__init__()
        
        # Define input layer to take in feature vector
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])  # input_dim here should match your features
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)]
        )
        
        # Define output layer for binary classification (background and signal)
        #self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.relu = nn.ReLU()                # Activation function
        self.sigmoid = nn.Sigmoid()          # Output activation


    def forward(self,x):
        # Concatenate mass parameter to feature vector
        #x = torch.cat(x, dim=1)
        
        # Forward pass through the input layer
        x = F.relu(self.input_layer(x))
        
        # Forward pass through the hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Output layer
        x = self.output_layer(x)
        
        return x.squeeze()


# In[15]:


# Initialize the model
input_dim = 14
model = ParametrizedDNN(input_dim=input_dim)



total = len(signal_dataset) + len(background_dataset)
weight_for_0 = (1 / len(background_dataset))*(total)/2.0
weight_for_1 = (1 / len(signal_dataset) )*(total)/2.0
#weight_for_2 = (1 / bckg)*(total)/3.0
#class_weight = {0: weight_for_0, 1: weight_for_1}
class_weight = torch.tensor([weight_for_0, weight_for_1])
print(class_weight)


# Define loss function and optimizer
#criterion = nn.CrossEntropyLoss(weight = class_weight)
criterion = nn.CrossEntropyLoss(weight = class_weight)
#criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = {'loss': []}  # Store the loss for each epoch


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for bno, batch in enumerate(balanced_loader):
        features = {key: batch[key] for key in batch if key != "label"}
        #features = torch.Tensor([v for k,v in features_dict.items()])
        labels = batch["label"].float()
        #print(labels)
        if bno>5:
            break
        # Move data to device
        #features, labels, mass_params = features.to(device), labels.to(device), mass_params.to(device)
        feature_tensors = [features[key].float() for key in features]  # List of tensors
        features_combined = torch.cat(feature_tensors, dim=0)  # Concatenate along the last dimension (typically the feature dimension)
        features_combined = features_combined.reshape(32,14)
        #print(features_combined.shape)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass with mass parameter
        outputs = model(features_combined)
        
        # Compute loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    # Calculate average loss for the epoch
    #avg_loss = total_loss / len(balanced_loader)  # Average over all batches
    avg_loss = total_loss   # Average over all batches
    history['loss'].append(avg_loss)  # Store average loss for plotting

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(balanced_loader)}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




