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
import datetime
from datetime import date, timedelta
from matplotlib import pyplot as plt
import math
import torch
import scipy.io
import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from tqdm import tqdm
from scipy.interpolate import interp1d
import torch_geometric.nn as N
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool, global_mean_pool, GATConv
from torch_geometric.data import Data, Batch, TemporalData, DataLoader
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, StaticGraphTemporalSignalBatch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
#%% Test


# loader = ChickenpoxDatasetLoader()

# dataset = loader.get_dataset()

# train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# for time, snapshot in enumerate(train_dataset):
#     x = snapshot.x
#     edge_index = snapshot.edge_index
#     edge_attr = snapshot.edge_attr
#         # y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
#         # cost = cost + torch.mean((y_hat-snapshot.y)**2)
#%%
def checkCountries(c1,c2):
    for c in c1:
        if c in c2:
            continue
        else:
            print(c, difflib.get_close_matches(c, c2))
            
#%% Define adjacency matrix
routes = pd.read_csv(r"data/routes.dat", header = None) 
airports = pd.read_csv(r"data/airports.dat", header = None) #4th is country

#Country-AirportID
IATA_country = np.transpose(np.array([list(airports[3]), list(airports[0])]))
# ICAO_country = np.transpose(np.array([list(airports[3]), list(airports[5])]))

rou = np.transpose(np.array([list(routes[3]), list(routes[5])]))
route = np.empty(rou.shape, dtype=np.dtype('U100'))
for i in range(len(rou)):
    # if (rou[i][0]=='\\N') or (rou[i][1]=='\\N'):
    #     rou = np.delete(rou,i,0)
    #     route = np.delete(route,i,0)
    try:
        while(np.where(IATA_country[:,1]==rou[i][1])[0].size==0) or (np.where(IATA_country[:,1]==rou[i][0])[0].size==0) or (rou[i][0]=='\\N') or (rou[i][1]=='\\N'):
            rou = np.delete(rou,i,0)
            route = np.delete(route,i,0)
    except:
        break
        
    route[i][0] = str(IATA_country[np.where(IATA_country[:,1]==rou[i][0]),0][0][0])   
    route[i][1] = str(IATA_country[np.where(IATA_country[:,1]==rou[i][1]),0][0][0])
    
variants = pd.read_csv(r"data/global_vars_May22.csv") #3rd

# countries = pd.read_csv(r"countries.csv")
# all_vars = pd.read_csv(r"all_vars21.csv")

#Make sure countries are spelt the same between variants and airport dataset
# checkCountries(countries,IATA_country[:,0])
        
#United States->USA
route[:,0] = np.char.replace(route[:,0], "United States", "USA")
route[:,1] = np.char.replace(route[:,1], "United States", "USA")

#list(set(variants['country']))
with open(r"countries_clustered_top30.csv", mode = 'r') as file:
    reader = csv.reader(file)
    countries = []
    for row in reader:
        countries.append(row[0])

with open(r"all_vars21_clustered.csv", mode = 'r') as file:
    reader = csv.reader(file)
    all_variants = []
    for row in reader:
        all_variants.append(row[0])

all_variants = all_variants[1:]
print("\n \n \n Replaced....")

countires_NA= ["Union of the Comoros", "Liechtenstein", "Kosovo", "Timor-Leste"]
        

adj_mat = np.eye(len(countries))
for i in range(len(countries)):
    source_table = route[route[:,0] ==countries[i]]
    adj_mat[i, np.isin(countries,source_table[:,1])] = 1
        
# np.count_nonzero(adj_mat,1) #~20 no routes

#%% edge weights
pop = pd.read_csv(r"pops.csv")
popu = pop[pop['Country Name'].isin(countries)]

popu =torch.from_numpy(popu["2022"].values).float() 
print(popu)

weighted_mat = np.ones((len(countries),len(countries)))
   
count = 0     
for i in range(len(countries)):
    try:
        source_pop = pop[pop['Country Name'] == countries[i]]["2022"].values[0]
    except:
        continue
    for j in range(len(countries)):
        if i >= j:
            continue
        try:
            dest_pop = pop[pop['Country Name'] == countries[j]]["2022"].values[0]  
            # count = count + 1
            # print(count)
        except:
            continue
        temp = np.sqrt(source_pop*dest_pop)
        weighted_mat[i, j] = temp
        weighted_mat[j, i] = temp

    
#Mask the unnecessary ones  
weighted_mat = np.where(adj_mat == 0, 0, weighted_mat) 

# Normalize edge weights
min_val = np.sort(np.unique(weighted_mat.flatten()))[2]
max_val = np.max(weighted_mat)
# weighted_mat = weighted_mat/np.max(weighted_mat)     
weighted_mat[np.where(weighted_mat != 1)] = ((weighted_mat[np.where(weighted_mat != 1)] -min_val) / (max_val - min_val)) + 1
weighted_mat = np.where(adj_mat == 0, 0, weighted_mat) 


         
#%%
# assume adj_matrix is your adjacency matrix
adj_tensor = torch.tensor(adj_mat)

# find the non-zero indices in the adjacency matrix
rows, cols = torch.where(adj_tensor != 0)

# concatenate the row and column indices into a single matrix
edge_index = torch.stack([rows, cols], dim=0)

# (optional) convert edge_index to a long tensor if necessary
edge_index = edge_index.long()

# (optional) transpose edge_index to match the PyTorch Geometric format
edge_index = edge_index.transpose(0, 1)

weight_tensor = torch.tensor(weighted_mat)

edge_weights = []
for e in range(len(edge_index)):
    edge_idx = edge_index[e]
    i = edge_idx[0]
    j = edge_idx[1]
    edge_weights.append(weight_tensor[i][j])
    
edge_weights = torch.tensor(edge_weights)

#%% Load S and cap if needed
def remove_outliers(data):
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)].dropna()
    return clean_data
    
time_data = pd.read_csv(r"Processed_res21_1_clustered.csv")
S = pd.read_csv(r"growth_rates_clustered.csv")

time_data = time_data[time_data['country'].isin(countries)]
##Normalize data with S
#%% Define time-series data matrix
# df = time_data
# # Define the number of days to go back in the snapshot matrices
# T = 14

# # 0: never reaches prev, 1: hasn't reached prev, 2: reached prev
# temp_retro = "21J.Delta"

# Get the unique dates and pangoLineages in the DataFrame
def process_data(df,T):
    df['date'] = pd.to_datetime(df['date'])
    dates = df['date'].unique()# df['date'].unique()
    # pangoLineages = all_variants
    #df['pangoLineage'].unique()
    # date_0 = "2020-04-28";
    feat_mats = []#np.empty((0,len(countries),T+1))
    target_mats = []#np.array((0,len(countries),2))
    # Loop over each date and pangoLineage (each gives us a snapshot)
    # count = 0
    # batches = []
    # batchesT = []
    for d in dates:
        pangos = df[(df['date'] == d)]
        pangos = pangos['pangoLineage'].unique()
        #what if we have a ariant not present at a given dat, but is at other dates?
        # batch each varaint group  and dont do this maybe?
        for pangoLineage in pangos: #pangoLineages:
            p_index = all_variants.index(pangoLineage)
            si = S.S[p_index]
            # Create the feat_matrix and target_matrix
            feat_matrix = np.zeros((len(countries), T+1))
            target_matrix = np.zeros((len(countries), 2))
            target_matrix[:,0] = -1 #Will never reach prevalence
            countries_dom = df[(df['pangoLineage'] == pangoLineage) & (df['prev'] > 1/3)]
            
            k1 = countries_dom.drop_duplicates(subset='country', keep = 'first')
            idx2 = (k1.date - d).dt.days
            idx2[idx2 < 0] =0
            #Check if date of dom has passed, if it did then 0, else calculate it.
            idx = [countries.index(si) for si in countries_dom['country'].unique()]
            target_matrix[idx,0] = idx2
            target_matrix[idx,1] = 1 
            # Get the prev values for the current date and pangoLineage
            # prev_values = df[(pd.to_datetime(df['date']) >= pd.to_datetime(d) - pd.Timedelta(days=T-1)) &
            #                  (pd.to_datetime(df['date']) <= pd.to_datetime(d)) &
            #                  (df['pangoLineage'] == pangoLineage)]
            
            # prev_values = df[(df['Day'] >= (d) - (T-1)) &
            #                  (df['Day'] <= d) &
            #                  (df['pangoLineage'] == pangoLineage)]
            prev_values = df[(df['date'] >= (pd.Timestamp(d) - pd.DateOffset(days = T-1))) &
                             (df['date'] <= d) &
                             (df['pangoLineage'] == pangoLineage)]
            countries_pres = prev_values['country'].unique()
            # what about countries which eventually get filtered out
            if len(countries_pres)==0:
                continue
            for c in countries_pres:
                prev_values_c = prev_values[(prev_values['country'] == c)]['prev'].values
                # If no prev values were found, fill the row with 0s
                if len(prev_values_c) == 0:
                    prev_values_c = np.zeros(T)
                # If not enough prev values were found, pad with 0s
                elif len(prev_values_c) < T:
                    prev_values_c = np.pad(prev_values_c, (T-len(prev_values_c), 0), 'constant')
            
                prev_values_c = np.append(prev_values_c, si)    
                # Set the row in the feat_matrix to the prev values
                row_index = countries.index(c) #np.where(pangoLineages == pangoLineage)[0][0]
                feat_matrix[row_index, :] = np.log(prev_values_c +(10**-10))
                
                
                # Get the days_to_prev value for the current date and pangoLineage
                target_vals = prev_values[(prev_values['date'] == d) & (prev_values['country'] == c)]
                
                days_to_prev = target_vals['days_to_prev'].values
                dom =  target_vals['B'].values
                # If no days_to_prev value was found, set it to 10000 (doesn't become prevalent)
                if len(days_to_prev) == 0:
                    continue
                
                # Set the value in the target_matrix to the days_to_prev value
                target_matrix[row_index, 0] = days_to_prev
                target_matrix[row_index, 1] = dom
                
                
            # snap = Data(x = feat_matrix, edge_index = edge_index, y = target_matrix)
            feat_mats.append(feat_matrix)# = np.append(feat_mats,feat_matrix, axis = 0)
            target_mats.append(target_matrix) # = np.append(target_mats, target_matrix, axis = 0)
    
    dataset = StaticGraphTemporalSignal(edge_index = edge_index.numpy().T, edge_weight = edge_weights.numpy().T,
                                        features = feat_mats, targets = target_mats)
    
    return dataset
    # batchesF.append(feat_mats)
    # batchesT.append(target_mats)
    # feat_mats=[]
    # target_mats=[]
        # torch.save(snap, f'snap_data/snap_{count}.pt')
        # count = count + 1
        
def process_data_test(df,T,d):
    df['date'] = pd.to_datetime(df['date'])
    # dates = df['date'].unique()# df['date'].unique()
    # pangoLineages = all_variants
    #df['pangoLineage'].unique()
    # date_0 = "2020-04-28";
    
    #Here get the passed 14 days of data (interpolated features)
    
    feat_mats = []#np.empty((0,len(countries),T+1))
    target_mats = []#np.array((0,len(countries),2))
    # Loop over each date and pangoLineage (each gives us a snapshot)
    # count = 0
    # batches = []
    # batchesT = []
    # pangos = df[(df['date'] == d)]
    pangos = df['pangoLineage'].unique()
    #what if we have a ariant not present at a given dat, but is at other dates?
    # batch each varaint group  and dont do this maybe?
    for pangoLineage in pangos: #pangoLineages:
        p_index = all_variants.index(pangoLineage)
        si = S.S[p_index]
        # Create the feat_matrix and target_matrix
        feat_matrix = np.zeros((len(countries), T+1))
        target_matrix = np.zeros((len(countries), 2))
        target_matrix[:,0] = -1 #Will never reach prevalence
        countries_dom = df[(df['pangoLineage'] == pangoLineage) & (df['prev'] > 1/3)]
        
        k1 = countries_dom.drop_duplicates(subset='country', keep = 'first')
        idx2 = (k1.date - d).dt.days
        idx2[idx2 < 0] =0
        #Check if date of dom has passed, if it did then 0, else calculate it.
        idx = [countries.index(si) for si in countries_dom['country'].unique()]
        target_matrix[idx,0] = idx2
        target_matrix[idx,1] = 1 
        # Get the prev values for the current date and pangoLineage
        # prev_values = df[(pd.to_datetime(df['date']) >= pd.to_datetime(d) - pd.Timedelta(days=T-1)) &
        #                  (pd.to_datetime(df['date']) <= pd.to_datetime(d)) &
        #                  (df['pangoLineage'] == pangoLineage)]
        
        # prev_values = df[(df['Day'] >= (d) - (T-1)) &
        #                  (df['Day'] <= d) &
        #                  (df['pangoLineage'] == pangoLineage)]
        prev_values = df[(df['date'] >= (pd.Timestamp(d) - pd.DateOffset(days = T))) &
                         (df['date'] <= d) &
                         (df['pangoLineage'] == pangoLineage)]
        countries_pres = prev_values['country'].unique()
        # what about countries which eventually get filtered out
        if len(countries_pres)==0:
            continue
        for c in countries_pres:
            prev_values_c = prev_values[(prev_values['country'] == c)]['prev'].values
            # If no prev values were found, fill the row with 0s
            if len(prev_values_c) == 0:
                prev_values_c = np.zeros(T)
            # If not enough prev values were found, pad with 0s
            elif len(prev_values_c) == 1:
                # interpolate from 0
                func = interp1d([0, 14],[0, prev_values_c[0]])
                x = [h for h in range(1,15)]
                prev_values_c = func(x)
                # prev_values_c = np.pad(prev_values_c, (T-len(prev_values_c), 0), 'constant')
            elif len(prev_values_c) == 2:
                # interpolate between the two points (no need for cumilative since not by case, by prev)
                func = interp1d([0, 14],prev_values_c)
                x = [h for h in range(1,15)]
                prev_values_c = func(x)
        
            prev_values_c = np.append(prev_values_c, si)    
            # Set the row in the feat_matrix to the prev values
            row_index = countries.index(c) #np.where(pangoLineages == pangoLineage)[0][0]
            feat_matrix[row_index, :] = np.log(prev_values_c +(10**-10))
            
            
            # Get the days_to_prev value for the current date and pangoLineage
            target_vals = prev_values[(prev_values['date'] == d) & (prev_values['country'] == c)]
            
            days_to_prev = target_vals['days_to_prev'].values
            dom =  target_vals['B'].values
            # If no days_to_prev value was found, set it to 10000 (doesn't become prevalent)
            if len(days_to_prev) == 0:
                continue
            
            # Set the value in the target_matrix to the days_to_prev value
            target_matrix[row_index, 0] = days_to_prev
            target_matrix[row_index, 1] = dom
            
            
        # snap = Data(x = feat_matrix, edge_index = edge_index, y = target_matrix)
        feat_mats.append(feat_matrix)# = np.append(feat_mats,feat_matrix, axis = 0)
        target_mats.append(target_matrix) # = np.append(target_mats, target_matrix, axis = 0)
    
    dataset = StaticGraphTemporalSignal(edge_index = edge_index.numpy().T, edge_weight = popu,
                                        features = feat_mats, targets = target_mats)
    
    return dataset
#%%
# df = time_data
# # Define the number of days to go back in the snapshot matrices
# T = 14
# dataset = process_data(df,T)#StaticGraphTemporalSignal(edge_index = edge_index.numpy().T, edge_weight = edge_index.numpy().T,
#                                    # features = feat_mats, targets = target_mats)


# dataset = StaticGraphTemporalSignalBatch(edge_index = edge_index.numpy().T, edge_weight = edge_index.numpy().T,
#                                     features = feat_mats, targets = target_mats, batches = 32)
# torch.save(dataset, f'processed_data.pt')


#%%
# loader = ChickenpoxDatasetLoader()

# # dataset = loader.get_dataset()

# train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
# train_dataset, val_dataset = temporal_signal_split(train_dataset, train_ratio=0.8)
#%% Define GNN
class GCN(torch.nn.Module):
    def __init__(self,node_features):
        super(GCN,self).__init__()
        self.conv1 = GCNConv(node_features, 32)
        self.conv2 = GCNConv(32, 16)
        # self.norm = torch.nn.LayerNorm(32)
        self.fc1 = torch.nn.Linear(16, 1)
        self.fc2 = torch.nn.Linear(33, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.conv1(x, edge_index, edge_weights)
        x = F.relu(x1)
        x = F.dropout(x, training=self.training)
        x2 = self.conv2(x, edge_index, edge_weights)
        # print(x.shape)
        #Might need normalization here since we concatenate a value between 0 and 1
        # x = self.norm(x)
        # x1 = self.conv2(x, edge_index)
        x1 = self.fc1(x2)
        c = F.sigmoid(x1) #Need class weights
        x = torch.cat([x,c],dim = 1)
        # print(x.shape)
        # print(c.shape)
        
        r = self.fc2(x)
        # r = F.relu(r0)
        # mask = c < 0.5
        # r[mask] = -1
        # r = F.relu(r)
        return r,x1
    
class GIN(torch.nn.Module):
    def __init__(self,node_features):
        super(GIN,self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(node_features,32),
                       BatchNorm1d(32), ReLU(),
                       Linear(32,32), ReLU()))
        self.conv2 =  GINConv(
            Sequential(Linear(32,32),
                       BatchNorm1d(32), ReLU(),
                       Linear(32,32), ReLU()))
        self.conv3 =  GINConv(
            Sequential(Linear(32,32),
                       BatchNorm1d(32), ReLU(),
                       Linear(32,1), ReLU()))
        # self.norm = torch.nn.LayerNorm(32)
        self.fc1 = torch.nn.Linear(4, 32)      
        self.fc2 = torch.nn.Linear(32, 1)  
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.conv1(x, edge_index)
        # x = F.relu(x1)
        # x = F.dropout(x, training=self.training)
        x2 = self.conv2(x1, edge_index)
        # print(x.shape)
        x3 = self.conv3(x2, edge_index)
        #Might need normalization here since we concatenate a value between 0 and 1
        # x = self.norm(x)
        # x1 = self.conv2(x, edge_index)
        # x1 = global_add_pool(x1, data.batch)
        # x2 = global_add_pool(x2, data.batch)
        # x3 = global_add_pool(x3, data.batch)
        # x1 = N.aggr.SumAggregation(x1, dim = 0)
        # x2 = N.aggr.SumAggregation(x2, dim = 0)
        # x3 = N.aggr.SumAggregation(x3, dim = 0)
        # print(x3.shape, x1.shape, x2.shape)
        x1 = torch.sum(x1, dim = 1, keepdim = True)
        x2 = torch.sum(x2, dim = 1, keepdim = True)
        x3 = torch.sum(x3, dim = 1, keepdim = True)
        # print(x3.shape, x1.shape, x2.shape)
        
        c = F.sigmoid(x3) #Need class weights
        h = torch.cat((x1,x2,x3), dim = 1)
        # print(h.shape)
        # print(c.shape)
        x = torch.cat([h,c],dim = 1)
        # print(x.shape)
        # print(c.shape)
        
        r = self.fc1(x)
        r = r.relu()
        r = F.dropout(r, p=0.5, training = self.training)
        r = self.fc2(r)
        # r = F.relu(r0)
        # mask = c < 0.5
        # r[mask] = -1
        # r = F.relu(r)
        return r,x3

class GAT(torch.nn.Module):
    def __init__(self,node_features,heads):
        super(GAT,self).__init__()
        self.conv1 = GATConv(node_features, 32,heads, dropout=0.4)
        self.conv2 = GATConv(32*heads, 16, heads,concat=False)
        # self.norm = torch.nn.LayerNorm(32)
        self.fc1 = torch.nn.Linear(16*heads, 1)
        self.fc2 = torch.nn.Linear(33, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        x = F.dropout(x, training=self.training)
        x2 = self.conv2(x, edge_index)
        # print(x.shape)
        #Might need normalization here since we concatenate a value between 0 and 1
        # x = self.norm(x)
        # x1 = self.conv2(x, edge_index)
        x1 = self.fc1(x2)
        c = F.sigmoid(x1) #Need class weights
        x = torch.cat([x,c],dim = 1)
        # print(x.shape)
        # print(c.shape)
        
        r = self.fc2(x)
        # r = F.relu(r0)
        # mask = c < 0.5
        # r[mask] = -1
        # r = F.relu(r)
        return r,x1


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        r = F.relu(h) #r
        c = F.sigmoid(h)
        h = torch.cat([r,c],dim = 1)
        h = self.linear(h)
        h = F.relu(h)
        return h
        
        
        
class GCN_MLP(torch.nn.Module):
    def __init__(self,node_features, input_size, output_size):
        super(GCN_MLP,self).__init__()
        self.fc11 = torch.nn.Linear(input_size, 512)  # Fully connected layer 1
        self.fc22 = torch.nn.Linear(512, output_size)  # Fully connected layer 2
        self.conv1 = GCNConv(node_features, 32)
        self.conv2 = GCNConv(32, 16)
        # self.norm = torch.nn.LayerNorm(32)
        self.fc1 = torch.nn.Linear(16, 1)
        self.fc2 = torch.nn.Linear(33, 1)


    def forward(self, data, population):
        x, edge_index = data.x, data.edge_index
        
        edge_weights = self.fc11(population)  # Assuming 'population' is a tensor containing population information
        edge_weights = self.fc22(edge_weights)
        edge_weights = torch.sigmoid(edge_weights)
        
        x1 = self.conv1(x, edge_index, edge_weights)
        x = F.relu(x1)
        x = F.dropout(x, training=self.training)
        x2 = self.conv2(x, edge_index, edge_weights)
        # print(x.shape)
        #Might need normalization here since we concatenate a value between 0 and 1
        # x = self.norm(x)
        # x1 = self.conv2(x, edge_index)
        x1 = self.fc1(x2)
        c = F.sigmoid(x1) #Need class weights
        x = torch.cat([x,c],dim = 1)
        # print(x.shape)
        # print(c.shape)
        
        r = self.fc2(x)
        # r = F.relu(r0)
        # mask = c < 0.5
        # r[mask] = -1
        # r = F.relu(r)
        return r,x1
    
class EarlyStopper:
    def __init__(self, patience = 1, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = np.inf
        self.weights = []
    
    def early_stop(self,validation_loss,weights):
        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
            self.counter = 0
            self.weights = [] # clear old weights and save new ones
            for param in weights:
                self.weights.append(param.data)
            # self.weights = weights
        elif validation_loss > (self.min_val_loss + self.min_delta):
            self.counter +=1
            if self.counter >= self.patience:
                return True
        return False
            
#%% Train GNN
def train(T,epochs,optimizer,early_stopper,weight_pos, popu = None):
    torch.cuda.empty_cache()
    for epoch in tqdm(range(epochs)):
        model.train()
        cost_1 = torch.tensor(0).to(device)
        cost = torch.tensor(0).to(device)
        for time, snapshot in enumerate(train_dataset):
            if popu is not None:
              y_hat = model(snapshot, popu)
            else:
              y_hat = model(snapshot)#.x.to(device), snapshot.edge_index.to(device), snapshot.edge_attr.to(device))
            cost_1 = F.binary_cross_entropy_with_logits(y_hat[1].squeeze(), snapshot.y[:,1],pos_weight=(weight_pos))
            cost_2 = torch.mean((y_hat[0].squeeze()-snapshot.y[:,0].to(device))**2)
            cost = cost + cost_2 + cost_1

        print(time)
        cost = cost / (time+1)
        print(cost.item())
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        val_cost = torch.tensor(0).to(device)
        if epoch < 5:
            continue
        with torch.no_grad():
            for time, snapshot in enumerate(val_dataset):
                if popu is not None:
                  y_hat = model(snapshot, popu)
                else:
                  y_hat = model(snapshot)
                # val_cost = val_cost + torch.mean(torch.abs((y_hat-snapshot.y)))
                # cost_1 =  F.binary_cross_entropy(y_hat[1].squeeze(), snapshot.y[:,1],weight = weight_class)#loss_1(y_hat[1].squeeze(), snapshot.y[:,1])
                cost_1 = F.binary_cross_entropy_with_logits(y_hat[1].squeeze(), snapshot.y[:,1],pos_weight=(weight_pos))
                val_cost = val_cost + torch.mean((y_hat[0].squeeze()-snapshot.y[:,0].to(device))**2) + cost_1
            val_cost = val_cost / (time+1)
            val_cost = val_cost.item()
            print("Val _cost: {:.4f}".format(val_cost))
            if early_stopper.early_stop(val_cost, model.parameters()):
                #Update with best weights and stop training
                count = 0
                best_weights = early_stopper.weights
                for param in model.parameters():
                    param.data = best_weights[count]
                    count +=1
                print("Early stop, best Val_Loss: ", early_stopper.min_val_loss)
                return
        
#%% Test GNN
def eval_F1_MAE(popu = None):
    model.eval()
    cost_1 = torch.tensor(0).to(device)
    cost = torch.tensor(0).to(device)
    CF = 0
    count = 0
    cost_median = 0
    with torch.no_grad():
        for time, snapshot in enumerate(test_dataset): 
            if popu is not None:
              y_hat = model(snapshot, popu)
            else:
              y_hat = model(snapshot)
            pred = F.sigmoid(y_hat[1])
            CF = CF + confusion_matrix(snapshot.y[:,1].detach().numpy(), np.round((pred.squeeze().detach().numpy())))
            cost_1 = cost_1 + f1_score(snapshot.y[:,1].detach().numpy(), np.round((pred.squeeze().detach().numpy())), average = 'weighted')
            # correct_idx = np.bitwise_and(snapshot.y[:,1].detach().numpy() == np.round((pred.squeeze().detach().numpy())), (snapshot.y[:,1].detach().numpy() == 1))
            correct_idx = (snapshot.y[:,1].detach().numpy() == 1)
            correct_idx2 = (snapshot.y[:,0].detach().numpy() != 0)
            correct_idx = np.bitwise_and(correct_idx,correct_idx2)
            # correct_idx = np.bitwise_and(correct_idx,snapshot.y[:,0].detach().numpy() !=0)
            #y_hat has to be > and a multiple of 14
            if np.any(correct_idx):
                count += 1
                pred = y_hat[0].squeeze()[correct_idx]
                cost = cost + torch.mean(torch.abs(((np.ceil(np.maximum(pred,1)/14)*14)-snapshot.y[correct_idx,0])))
                cost_median = cost_median + torch.median(torch.abs(((np.ceil(np.maximum(pred,1)/14)*14)-snapshot.y[correct_idx,0])))
        cost_1 = cost_1 / (time+1)
        cost_1 = cost_1.item()
        if count !=0:           
            cost = cost / (count)
            cost_median = cost_median/ count       
            cost = cost.item()
            cost_median =  cost_median.item()
        else:
            cost = -1
            cost_median = -1
        
        print("CF", CF)
        
        print("average F1: {:.4f}".format(cost_1))
        
        print("average MAE: {:.4f}".format(cost))
        
        print("average median AE:", cost_median)
        return CF,cost_1,cost,cost_median,pred
    #%% class weights
# all_labels = np.concatenate(train_dataset.targets)
# classes, classes_counts = np.unique(all_labels[:,1], return_counts = True)

# weight_class = torch.tensor(len(all_labels)/(len(classes)*classes_counts))
# weight_pos = weight_class[1]# torch.tensor(classes_counts[0]/classes_counts[1])
# #%% test pipe
# # torch.cuda.empty_cache()
# # wks_back = 2
# batch_size = 32 # train_size/nb of days for train?
# epochs = 100
# device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu') #cpu 2:30
# model = GCN(node_features = T + 1).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# early_stopper = EarlyStopper(patience=4,min_delta=5)
# train(T,epochs,optimizer,early_stopper, weight_pos)

# CF,f1,MAE = eval_F1_MAE()
#%% Retro 2 evals
# df = time_data
T = 14

#Load GT
data_GT = pd.read_csv(r"Processed_res21_1_clusteredGT.csv")
data_GT = data_GT[data_GT['country'].isin(countries)]

#Choose a variant
temp_retro = "22F.Omicron"

# #Find date of first appearance & date of last appearance
time_data['date'] = pd.to_datetime(time_data['date'])
data_GT['date'] = pd.to_datetime(data_GT['date'])

temp = data_GT[data_GT['pangoLineage'] == temp_retro]
dates = temp['date'].unique()
dates.sort()

CF = []
MAE = []
med_MAE = []
F1 = []
#Loop over these 2 dates (every 2 weeks):
 # For each date d, train & validate on the interpolated data up till d-1
for d in dates: # retrospective dates
    df = time_data[time_data['date']< (d - pd.Timedelta(days=T-1))] #Retro TRAINING data
    # df_GT = data_GT[data_GT['date']<=d]
    #One timestep or all timesteps for testing? && PER VARIANT
    df_GT_Te = data_GT[(data_GT['pangoLineage'] == temp_retro)] #Retro TESTING data (data_GT['date'] == d) & 
    if len(df_GT_Te) ==0:
        continue
        
    dataset = process_data(df,T) #StaticGraphTemporalSignal(edge_index = edge_index.numpy().T, edge_weight = edge_index.numpy().T,
                                            #features = feat_mats, targets = target_mats)

    # train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    #validate on approx last month of data
    100/dataset.snapshot_count
    train_dataset, val_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    
    test_dataset = process_data_test(df_GT_Te,T,d)
    #Find class weights
    all_labels = np.concatenate(train_dataset.targets)
    classes, classes_counts = np.unique(all_labels[:,1], return_counts = True)

    weight_class = torch.tensor(len(all_labels)/(len(classes)*classes_counts))
    
    weight_pos = weight_class[1]# torch.tensor(classes_counts[0]/classes_counts[1])
    
    torch.cuda.empty_cache()
    epochs = 100
    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu') #cpu 2:30
    print(edge_index.shape)
    model = GCN_MLP(node_features = T + 1,input_size = len(popu), output_size = edge_index.shape[0]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    early_stopper = EarlyStopper(patience=3,min_delta=5)
    train(T,epochs,optimizer,early_stopper, weight_pos,popu)

    # Evaluate on date d USING GT:
        # Classifier f1 evaluation
        # For correctly classified 1s, find MAE
        # save thes values in lists for later plotting
        
    CF1,f11,MAE1,MAE2,pred = eval_F1_MAE(popu)
    if MAE1 == -1:
        break
    MAE.append(MAE1)
    med_MAE.append(MAE2)
    F1.append(f11)
    CF.append(CF1)
    print(pred)
    
print("FINAL RES :", temp_retro)
print("F1: ", F1)
print("MEDIAN: ", med_MAE)
print("MAE: ", MAE)


