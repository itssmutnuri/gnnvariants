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
import copy
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
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool, global_mean_pool, GATConv, norm
from torch_geometric.data import Data, Batch, TemporalData, DataLoader  
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, StaticGraphTemporalSignalBatch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


from config import is_graph, ADJ_bool, Flights_bool, self_loops, EW_bool
from config import topX_bool, topX_C, dom_thresh, use_r, use_S
from config import min_epochs, max_epochs, T, reg_bool
from config import early_stopper_patience, early_stopper_delta
from config import variants_path, countries_path, device, ITERATION_NAME
from config import IS_DEBUG

from modulefuncs.edge_weight_calc import edgeW_calc
from modulefuncs.data_processing import process_data, process_data_test
from modulefuncs.utils import append_to_csv
from modulefuncs.models import EarlyStopper, ModelM

from functools import partial

import path_values as paths


if topX_bool:
  nb_of_countries = topX_C
else:
  nb_of_countries = 0

if dom_thresh == 1/9:
    data_path = paths.PATH_PROCESSED_THRES_1_9th
elif dom_thresh == 0.25:
    data_path = paths.PATH_PROCESSED_THRES_0_25
elif dom_thresh == 1/3:
    data_path = paths.PATH_PROCESSED_THRES_1_3
elif dom_thresh == 0.5:
    data_path = paths.PATH_PROCESSED_THRES_0_5
elif dom_thresh == 1:
    data_path = paths.PATH_PROCESSED_THRES_1
else:
  raise ValueError("Invalid option. Please choose a valid option for dom_thresh: [1/9,0.25,1/3,0.5,1].")            
            
            
#Load GT data
time_data = pd.read_csv(data_path)

data_GT = pd.read_csv(data_path)
restrictions = pd.read_csv(paths.PATH_TRAVEL_CONTROLS)

#Get a list of variants to evaluate on
all_variants = data_GT['pangoLineage'].unique().tolist()

with open(countries_path, mode = 'r') as file: #Load countries
    reader = csv.reader(file)
    countries = []
    for row in reader:
        countries.append(row[0])

with open(variants_path, mode = 'r') as file: #load variants
    reader = csv.reader(file)
    variant_names = []
    for row in reader:
        variant_names.append(row[0])

variant_names = variant_names[1:]
print("\n \n \n Replaced....")

countires_NA= ["Union of the Comoros", "Liechtenstein", "Kosovo", "Timor-Leste"]
countries = countries[1:]


#Use only top X countries
if nb_of_countries !=0:
  filtered_country_names = []
  top_X_countries = time_data['country'].value_counts().nlargest(nb_of_countries).index.tolist()

  # Now, we have a list of the top X countries with the most data
  time_data = time_data[time_data['country'].isin(top_X_countries)]
  data_GT = data_GT[data_GT['country'].isin(top_X_countries)]
  for country in countries:
    if country in top_X_countries:
        filtered_country_names.append(country)
  
  countries = filtered_country_names

if is_graph:            
  # Define adjacency matrix
  routes = pd.read_csv(paths.PATH_ROUTES, header = None) 
  airports = pd.read_csv(paths.PATH_AIRPORTS, header = None) #4th is country
  
  #Country-AirportID
  IATA_country = np.transpose(np.array([list(airports[3]), list(airports[0])]))
  
  rou = np.transpose(np.array([list(routes[3]), list(routes[5])]))
  route = np.empty(rou.shape, dtype=np.dtype('U100'))
  for i in range(len(rou)):
      try:
          while(np.where(IATA_country[:,1]==rou[i][1])[0].size==0) or (np.where(IATA_country[:,1]==rou[i][0])[0].size==0) or (rou[i][0]=='\\N') or (rou[i][1]=='\\N'):
              rou = np.delete(rou,i,0)
              route = np.delete(route,i,0)
      except:
          break
          
      route[i][0] = str(IATA_country[np.where(IATA_country[:,1]==rou[i][0]),0][0][0])   
      route[i][1] = str(IATA_country[np.where(IATA_country[:,1]==rou[i][1]),0][0][0])
      
  
          
  #United States->USA
  route[:,0] = np.char.replace(route[:,0], "United States", "USA")
  route[:,1] = np.char.replace(route[:,1], "United States", "USA")
  
  
  if self_loops:
    adj_mat = np.eye(len(countries)) #Enables internal pressure
  else:
    adj_mat = np.zeros((len(countries),len(countries)))
  
  
  with open(paths.PATH_COUNTRY_ADJ, 'r') as json_file:
    data_json = json.load(json_file)
    for i in range(len(countries)):
        #Takes care of islands which dont have adjacencies
        if countries[i] == "Aruba" or countries[i] == "Fiji" or countries[i] == "Guadeloupe" or countries[i] == "Iceland" or countries[i] == "Jamaica" or countries[i] == "Maldives" or countries[i] == "Mauritius" or countries[i] == "New Zealand" or countries[i] =="Seychelles":
          C = []
        elif countries[i] == "Hong Kong": 
          C= ["China"]
        elif countries[i] == "Russia":
          C = data_json["Russian Federation"]
        elif countries[i] =="USA":
          C = data_json["United States of America"]
        else:
          C = data_json[countries[i]]
        
        if Flights_bool:
          source_table = route[route[:,0] ==countries[i]]
          adj_mat[i, np.isin(countries,source_table[:,1])] = 1
          
        if ADJ_bool:
          for c in C:
            adj_mat[i, np.isin(countries,c)] = 1
  
  # adj_mat = np.ones(adj_mat.shape) #UNCOMMENTING MAKES GRAPH FULLY CONNECTED
  adj_tensor = torch.tensor(adj_mat)
  
  # find the non-zero indices in the adjacency matrix
  rows, cols = torch.where(adj_tensor != 0)
  
  # concatenate the row and column indices into a single matrix
  edge_index = torch.stack([rows, cols], dim=0)
  
  # convert edge_index to a long tensor if necessary
  edge_index = edge_index.long()
  
  # transpose edge_index to match the PyTorch Geometric format
  edge_index = edge_index.transpose(0, 1)

# Load S values

S = pd.read_csv(paths.PATH_GROWTH_RATES)

# use partial to assign all args except the df value. this is done when processing.
partial_edgeW_calc = partial(edgeW_calc, countries=countries, adj_mat=adj_mat, EW_bool=EW_bool)


# Train model
def train(T,epochs,optimizer,early_stopper,weight_pos,edge_weights_T,edge_weights_V, reg = 1):
    torch.cuda.empty_cache()
    for epoch in tqdm(range(epochs)):
        model.train()
        cost_1 = torch.tensor(0).to(device)
        cost = torch.tensor(0).to(device)
        for time, snapshot in enumerate(train_dataset):
            EW = edge_weights_T[time]
            
            y_hat = model(snapshot,EW)
            
            if reg:
              y_hat = F.relu(y_hat)
              mask = snapshot.y[:,1].detach().numpy() == 1 #Create a mask for nodes to train on
              cost = cost + torch.mean((y_hat[mask].squeeze()-snapshot.y[mask,0].to(device))**2)
            else:
              cost = cost + F.binary_cross_entropy_with_logits(y_hat.squeeze(), snapshot.y[:,1],pos_weight=(weight_pos))

        print(time)
        cost = cost / (time+1)
        print(cost.item())
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch < min_epochs:
            continue
            
        model.eval()
        val_cost = torch.tensor(0).to(device)
        with torch.no_grad():
            for time, snapshot in enumerate(val_dataset):
                EW = edge_weights_V[time]
                if reg:
                  y_hat = model(snapshot,EW)
                  mask = snapshot.y[:,1].detach().numpy() == 1
                  val_cost = val_cost + torch.mean((y_hat[mask].squeeze()-snapshot.y[mask,0].to(device))**2)
                else:
                  y_hat = model(snapshot,EW, True)
                  val_cost = val_cost + F.binary_cross_entropy_with_logits(y_hat.squeeze(), snapshot.y[:,1],pos_weight=(weight_pos))
                  
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
                return best_weights
    return early_stopper.weights
        
# Test model
def eval_F1_MAE(edge_weights_Te):
    model.eval()

    cost_1 = torch.tensor(0).to(device)
    cost = torch.tensor(0).to(device)
    CF = 0
    count = 0
    cost_median = 0
    with torch.no_grad():
        for time, snapshot in enumerate(test_dataset): #Currently, this is just for 1 timestep. Ca be done for more.
            EW = edge_weights_Te[time]

            correct_idx = (snapshot.y[:,1].detach().numpy() == 1) #Does become prevalent in a given country
            correct_idx2 = (snapshot.y[:,0].detach().numpy() != 0) #Is not already prevlanet in a given country 
            correct_idx = np.bitwise_and(correct_idx,correct_idx2)
            pred = np.array([])
            if np.any(correct_idx):
              count += 1
              if not reg_bool:
                y_hat = model(snapshot,EW)
                pred = F.sigmoid(y_hat) #predict for all nodes
                CF = CF + confusion_matrix(snapshot.y[:,1].detach().numpy(), np.round((pred.squeeze().detach().numpy())))
                cost_1 = cost_1 + f1_score(snapshot.y[:,1].detach().numpy(), np.round((pred.squeeze().detach().numpy())), average = 'macro')
              
              else:             #y_hat has to be > and a multiple of 14                
                y_hat = model(snapshot,EW)
                pred = y_hat.squeeze()[correct_idx] #Predict only for nodeswhich will become dominant
                cost = cost + torch.mean(torch.abs(((np.ceil(np.maximum(pred,1)/14)*14)-snapshot.y[correct_idx,0])))
                cost_median = cost_median + torch.median(torch.abs(((np.ceil(np.maximum(pred,1)/14)*14)-snapshot.y[correct_idx,0])))
        if count !=0:
            if reg_bool:           
              cost = cost / (count)
              cost_median = cost_median/ count       
              cost = cost.item()
              cost_median =  cost_median.item()
              CF = -1
              cost_1 = -1
            else:
              cost = -1
              cost_median = -1
              cost_1 = cost_1 / (count)
              cost_1 = cost_1.item()
              CF = CF / (count)
        else:
            cost = -1
            cost_median = -1
            cost_1 = -1
            CF = -1
        
        print("CF", CF)
        
        print("average F1: {:.4f}".format(cost_1))
        
        print("average MAE: {:.4f}".format(cost))
        
        print("average median AE:", cost_median)
        
        return CF,cost_1,cost,cost_median,pred,correct_idx

# Generate the current timestamp for the entire run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
header = ["CF1", "f11", "MAE1", "MAE2", "pred", "date", "countries"]


PARENT_FOLDER = "Results"
SUB_FOLDER = f"{ITERATION_NAME}_{timestamp}"

ERROR_FILE = 'status.csv'


for variant in variant_names:

    print(f" ======================== RUNNING FOR {variant} ======================== ")

    # Format the directory and filename for the current lineage
    directory = f"{PARENT_FOLDER}/{SUB_FOLDER}"
    filename = f"{variant}.csv"
    filepath = os.path.join(directory, filename)
    status_filepath = os.path.join(directory, ERROR_FILE)
    
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # #Find date of first appearance & date of last appearance
    time_data['date'] = pd.to_datetime(time_data['date'])
    data_GT['date'] = pd.to_datetime(data_GT['date'])
    restrictions['date'] = pd.to_datetime(restrictions['date'])
    
    temp = data_GT[data_GT['pangoLineage'] == variant]
    dates = temp['date'].unique()
    dates.sort()

    # Loop between these 2 dates (biweekly):
    # For each date d, train & validate on the data up till d-1, and evaluate at d
    
    start_weights= []

    for d in dates: # retrospective dates
        try:

            df = time_data[time_data['date']< (d - pd.Timedelta(days=T-1))] #Retro TRAINING data
            
            df_GT_Te = data_GT[(data_GT['pangoLineage'] == variant)] #Retro TESTING data. test preprocess function given d as well
            
            if len(df_GT_Te) ==0:
                continue

            dataset, edgeWeights = process_data(df = df, T = T, 
                                                countries=countries, all_variants=all_variants,
                                                restrictions=restrictions, edge_index = edge_index,
                                                is_graph = is_graph, use_r=use_r, use_S=use_S, 
                                                dom_thresh=dom_thresh, partial_edgeW_calc=partial_edgeW_calc)

            
            #validate on 20% most recent data
            
            train_dataset, val_dataset = temporal_signal_split(dataset, train_ratio=0.8)
            LL = len(train_dataset.targets)
            train_edges = edgeWeights[:LL]
            val_edges = edgeWeights[LL:]

            test_dataset, edgeWeightsTE = process_data_test(df = df_GT_Te, T = T, d = d,
                                                            countries=countries, all_variants=all_variants,
                                                            restrictions=restrictions, edge_index=edge_index,
                                                            is_graph=is_graph, use_r=use_r, use_S=use_S,
                                                            dom_thresh=dom_thresh, partial_edgeW_calc=partial_edgeW_calc)

            #Find class weights
            if not reg_bool:
              all_labels = np.concatenate(train_dataset.targets)
              classes, classes_counts = np.unique(all_labels[:,1], return_counts = True)
  
              weight_class = torch.tensor(len(all_labels)/(len(classes)*classes_counts))
              
              weight_pos = weight_class[1]
            else:
              weight_pos = []
            
            torch.cuda.empty_cache()
            if use_S:
              model = ModelM(node_features = T+1).to(device)

            else:
              model = ModelM(node_features = T).to(device)
              
            if len(start_weights) != 0:              
              param_iter = iter(model.parameters())
              for weight_tensor in start_weights:
                  param = next(param_iter)
                  with torch.no_grad():
                      param.copy_(weight_tensor)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
            early_stopper = EarlyStopper(patience=early_stopper_patience,min_delta=early_stopper_delta)
            
            start_weights = train(T,max_epochs,optimizer,early_stopper, weight_pos,train_edges,val_edges, reg_bool)

            

            # Evaluate on date d USING GT:
            # Classifier f1 evaluation
            # For positives, find MAE
            # save thes values in lists

            CF1, f11, MAE1, MAE2, pred, country_mask = eval_F1_MAE(edgeWeightsTE)

            # Convert 1s and 0s to True and False. This shows us what countries we are predicting for at a given date d
            bool_country = [bool(val) for val in country_mask]
            selected_countries = list(compress(countries, bool_country))
            
            
            append_to_csv(filepath, [CF1, f11, MAE1, MAE2, pred.tolist(), d, selected_countries], header=header)
            
            print(pred)

            if (MAE1 == -1) and (f11==-1): #We have reached global dominance
                break
    
        except Exception as e:
            print(f"Error encountered for variant {variant}, Skipping.")
            append_to_csv(status_filepath, ['ERROR', variant, f'{e} with dates {d}'])
            continue
    
    append_to_csv(status_filepath, ['SUCCESS', variant, 'None'])

    if IS_DEBUG:
        break