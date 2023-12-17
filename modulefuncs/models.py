# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 11:55:10 2023

@author: user
"""

import numpy as np
import csv
import pandas as pd
import requests
import difflib
import torch_geometric as tg
from torch_geometric.utils import to_networkx
from itertools import compress
from datetime import date, timedelta, datetime
from matplotlib import pyplot as plt
import json
import os
import torch.optim.lr_scheduler as sch
# from csaps import csaps
import math
import torch
import scipy.io
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from tqdm import tqdm
from scipy.interpolate import interp1d
import torch_geometric.nn as N
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool, global_mean_pool, GATConv, norm
from torch_geometric.data import Data, Batch, TemporalData, DataLoader
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, StaticGraphTemporalSignalBatch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#%% Early stopper
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = np.inf
        self.weights = []

    def early_stop(self, validation_loss, weights):
        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
            self.counter = 0
            self.weights = []  # clear old weights and save new ones
            for param in weights:
                self.weights.append(param.data)
            # self.weights = weights
        elif validation_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
        
#%% MAIN MODEL DEF (MUST BE NAMED ModelM TO BE USED IN PIPELINE)
# Example GCN used
class ModelM(torch.nn.Module): ##IMPORT THIS
    def __init__(self,node_features):
        super(ModelM,self).__init__()
        self.conv1 = GCNConv(node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.norm1 = norm.GraphNorm(32)
        self.norm2 = norm.GraphNorm(16)
        self.fc1 = torch.nn.Linear(16, 1)


    def forward(self, data, edge_weight, norm = False):
        x, edge_index = data.x, data.edge_index

        x1 = self.conv1(x, edge_index, edge_weight)

        x = F.leaky_relu(x1)
        if norm:
          x = self.norm1(x)
        x = F.dropout(x, training=self.training)

        x2 = self.conv2(x.float(), edge_index, edge_weight.float())

        x2 = F.leaky_relu(x2)
        if norm:
          x2 = self.norm2(x2)

        x1 = self.fc1(x2)

        return x1

#%% OTHER MODEL DEFINITIONS.

#T-GCN
class GRU_GCN(torch.nn.Module):
    def __init__(self, node_features):
        super(GRU_GCN, self).__init__()

        self.conv1 = GCNConv(node_features+4, 8)
        # self.conv2 = GCNConv(16, 8)
        self.norm1 = norm.GraphNorm(node_features+4)
        self.norm2 = norm.GraphNorm(8)
        # self.norm3 = norm.GraphNorm(8)
        self.fc1 = torch.nn.Linear(8, 1)
        self.rnn = torch.nn.GRU(input_size=1, hidden_size=2, batch_first=True)

    def forward(self, data, edge_weight, norm=False):
        x, edge_index = data.x, data.edge_index
        x1 = (x[:, :4]).unsqueeze(2)

        x1, _ = self.rnn(x1)
        x1 = torch.cat((x1.flatten(1), x[:, -1].unsqueeze(1)), dim=1)
        x1 = self.norm1(x1)
        x1 = F.leaky_relu(x1)

        x2 = self.conv1(x1, edge_index, edge_weight)
        x2 = self.norm2(x2)
        x2 = F.leaky_relu(x2)
        x2 = F.dropout(x2, training=self.training, p=0.5)

        # x3 = self.conv2(x2.float(), edge_index, edge_weight)
        # x3 = self.norm3(x3)
        # x3 = F.leaky_relu(x3)

        x4 = self.fc1(x2.float())
        return x4

#GRU
class GRUs(torch.nn.Module):
    def __init__(self, node_features):
        super(GRUs, self).__init__()

        # self.conv1 = GCNConv(node_features+4, 8)
        # self.conv2 = GCNConv(16, 8)
        self.norm1 = norm.GraphNorm(node_features+4)
        # self.norm2 = norm.GraphNorm(8)
        # self.norm3 = norm.GraphNorm(8)
        self.fc1 = torch.nn.Linear(node_features+4, 1)
        self.rnn = torch.nn.GRU(input_size=1, hidden_size=2,batch_first=True)

    def forward(self, data, edge_weight, norm=False):
        x, edge_index = data.x, data.edge_index
        x1 = (x[:, :4]).unsqueeze(2)

        x1, _ = self.rnn(x1)
        x1 = torch.cat((x1.flatten(1), x[:, -1].unsqueeze(1)), dim=1)
        x1 = self.norm1(x1)
        x1 = F.leaky_relu(x1)
        x4 = self.fc1(x1.float())
        
        return x4


#MLP
class MLPs(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPs, self).__init__()
        
        # Input size includes both the sequential input and additional features
        self.input_size = input_size

        # Define the input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Define the output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        # Activation function (ReLU)
        self.activation = nn.ReLU()

    def forward(self, data, edge_weight, norm=False):
        # x is expected to be a tensor with shape (batch_size, sequence_length, input_size)
        
        # Reshape the input tensor to (batch_size * input_size)
        x = data.x
        x = (x[:, :4]).unsqueeze(2)
        
        # Forward pass through the layers
        x = self.activation(self.input_layer(x))
        
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
            
        output = self.output_layer(x)

        # Reshape the output tensor to (batch_size, output_size)
        output = output.view(-1, x.size(1), self.output_layer.out_features)

        return output