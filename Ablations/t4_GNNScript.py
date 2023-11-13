# -*- coding: utf-8 -*-
"""
Code with country adjacency and travel restrictions

Ablation: 
Set T = 4
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

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import path_values as paths

nb_of_countries = 0
time_data = pd.read_csv(paths.PATH_PROCESSED_DATA)

#Load GT
data_GT = pd.read_csv(paths.PATH_PROCESSED_DATA)
restrictions = pd.read_csv(paths.PATH_TRAVEL_CONTROLS)

with open(paths.PATH_COUNTRY_LIST, mode = 'r') as file:
    reader = csv.reader(file)
    countries = []
    for row in reader:
        countries.append(row[0])

with open(paths.PATH_VARIANT_LIST, mode = 'r') as file:
    reader = csv.reader(file)
    all_variants = []
    for row in reader:
        all_variants.append(row[0])

all_variants = all_variants[1:]
print("\n \n \n Replaced....")

countires_NA= ["Union of the Comoros", "Liechtenstein", "Kosovo", "Timor-Leste"]
countries = countries[1:]


if nb_of_countries !=0:
  filtered_country_names = []
  top_X_countries = time_data['country'].value_counts().nlargest(nb_of_countries).index.tolist()

  # Now, you have a list of the top 60 countries with the most data
  # You can use this list to filter your DataFrame, for example:
  time_data = time_data[time_data['country'].isin(top_X_countries)]
  data_GT = data_GT[data_GT['country'].isin(top_X_countries)]
  for country in countries:
    if country in top_X_countries:
        filtered_country_names.append(country)
  
  countries = filtered_country_names


def checkCountries(c1,c2):
    for c in c1:
        if c in c2:
            continue
        else:
            print(c, difflib.get_close_matches(c, c2))
            
# Define adjacency matrix
routes = pd.read_csv(paths.PATH_ROUTES, header = None) 
airports = pd.read_csv(paths.PATH_AIRPORTS, header = None) #4th is country

# Country-AirportID
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
    
variants = pd.read_csv(paths.PATH_VAR_22MAY) #3rd

        
#United States->USA
route[:,0] = np.char.replace(route[:,0], "United States", "USA")
route[:,1] = np.char.replace(route[:,1], "United States", "USA")

#list(set(variants['country']))

adj_mat = np.eye(len(countries))

#Aruba are islands and go to the else since they have no adj

with open(paths.PATH_COUNTRY_ADJ, 'r') as json_file:
  data_json = json.load(json_file)
  for i in range(len(countries)):
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
      for c in C:
        adj_mat[i, np.isin(countries,c)] = 1

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

# Load S and cap if needed
def remove_outliers(data):
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)].dropna()
    return clean_data
    

S = pd.read_csv(paths.PATH_GROWTH_RATES)

def edgeW_calc(df):
    weighted_mat = np.ones((len(countries),len(countries)))      

    # sort by restriction lvls 0->4
    df = df.sort_values(by='international_travel_controls')
    
    for c in df['country'].unique():
        original_value = df[df['country'] ==c]['international_travel_controls'].values[0]
        scaled_value = 1 - (original_value / 4) * 0.9
        indx = np.where(countries == c)
        weighted_mat[indx, :] = scaled_value
        weighted_mat[:,indx] = scaled_value
    
    #Mask the unnecessary ones  
    weighted_mat = np.where(adj_mat == 0, 0, weighted_mat) 
    for i in range(len(countries)):
      weighted_mat[i, i] = 1
    
    
    #print(weighted_mat)
    weight_tensor = torch.DoubleTensor(weighted_mat)
    
    edge_weights = []
    for e in range(len(edge_index)):
        edge_idx = edge_index[e]
        i = edge_idx[0]
        j = edge_idx[1]
        edge_weights.append(weight_tensor[i][j])
        
        
    #print(edge_weights.shape)
    edge_weights = torch.DoubleTensor(edge_weights)
    
    return edge_weights

# Get the unique dates and pangoLineages in the DataFrame
def process_data(df,T):
    dates = df['date'].unique()
   
    feat_mats = []
    target_mats = []
    edge_weights = []
    # Loop over each date and pangoLineage (each gives us a snapshot)
    for d in dates:
        pangos = df[(df['date'] == d)]
        pangos = pangos['pangoLineage'].unique()
        controls = restrictions[restrictions['date'] == d]
        
        EW = edgeW_calc(controls)

        # what if we have a ariant not present at a given dat, but is at other dates?
        # batch each varaint group  and dont do this maybe?
        for pangoLineage in pangos: #pangoLineages:
            p_index = all_variants.index(pangoLineage)
            si = S.S[p_index]

            # Create the feat_matrix and target_matrix
            feat_matrix = np.zeros((len(countries), T+1))
            target_matrix = np.zeros((len(countries), 2))
            target_matrix[:,0] = -1 # Will never reach prevalence
            countries_dom = df[(df['pangoLineage'] == pangoLineage) & (df['prev'] > 1/3)]
            
            k1 = countries_dom.drop_duplicates(subset='country', keep = 'first')
            idx2 = (k1.date - d).dt.days
            idx2[idx2 < 0] =0

            # Check if date of dom has passed, if it did then 0, else calculate it.
            idx = [countries.index(si) for si in countries_dom['country'].unique()]
            target_matrix[idx,0] = idx2
            target_matrix[idx,1] = 1

            # Get the prev values for the current date and pangoLineage
            prev_values = df[(df['date'] >= (pd.Timestamp(d) - pd.DateOffset(weeks = (T*2)-1))) &
                             (df['date'] <= d) &
                             (df['pangoLineage'] == pangoLineage)]
            countries_pres = prev_values['country'].unique()

            # what about countries which eventually get filtered out
            if len(countries_pres)==0:
                continue
            for c in countries_pres:
                temp_c = prev_values[(prev_values['country'] == c)].sort_values(by='date')
                two_week_intervals = pd.date_range(end=pd.Timestamp(d), periods=3, freq='-2W')
                temp_c = temp_c[temp_c['date'].isin(two_week_intervals)]
                prev_values_c = temp_c['prev'].values

                # If no prev values were found, fill the row with 0s
                if len(prev_values_c) == 0:
                    prev_values_c = np.zeros(T)

                # If not enough prev values were found, pad with 0s
                elif len(prev_values_c) < T:
                    prev_values_c = np.pad(prev_values_c, (T-len(prev_values_c), 0), 'constant')

                log_prev_vals = np.log(prev_values_c + (10**-10))
                appended_prev_si = np.append(log_prev_vals, si)
                row_index = countries.index(c)

                feat_matrix[row_index, : ] = appended_prev_si
                
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
                
            feat_mats.append(feat_matrix)
            target_mats.append(target_matrix)
            edge_weights.append(EW)
    dataset = StaticGraphTemporalSignal(edge_index = edge_index.numpy().T, edge_weight = edge_index.numpy().T,
                                        features = feat_mats, targets = target_mats)
    
    return dataset,edge_weights
        
def process_data_test(df,T,d):
    
    #Here get the passed 14 days of data (interpolated features)
    
    feat_mats = []
    target_mats = []
    # Loop over each date and pangoLineage (each gives us a snapshot)
    edge_weights = []
    pangos = df['pangoLineage'].unique()
    controls = restrictions[restrictions['date'] == d]
    EW = edgeW_calc(controls)

    # what if we have a ariant not present at a given dat, but is at other dates?
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
        prev_values = df[(df['date'] >= (pd.Timestamp(d) - pd.DateOffset(weeks = (T*2)-1))) &
                         (df['date'] <= d) &
                         (df['pangoLineage'] == pangoLineage)]
        countries_pres = prev_values['country'].unique()

        # what about countries which eventually get filtered out
        if len(countries_pres)==0:
            continue
        for c in countries_pres:
            temp_c = prev_values[(prev_values['country'] == c)].sort_values(by='date')
            two_week_intervals = pd.date_range(end=pd.Timestamp(d), periods=3, freq='-2W')
            temp_c = temp_c[temp_c['date'].isin(two_week_intervals)]
            prev_values_c = temp_c['prev'].values
            
            # If no prev values were found, fill the row with 0s
            if len(prev_values_c) == 0:
              prev_values_c = np.zeros(T)
            elif len(prev_values_c) < T:
              prev_values_c = np.pad(prev_values_c, (T-len(prev_values_c), 0), 'constant')
            
            log_prev_vals = np.log(prev_values_c + (10**-10))
            appended_prev_si = np.append(log_prev_vals, si)
            row_index = countries.index(c)

            feat_matrix[row_index, : ] = appended_prev_si
            
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
      
        feat_mats.append(feat_matrix)
        target_mats.append(target_matrix)
        edge_weights.append(EW)
    
    dataset = StaticGraphTemporalSignal(edge_index = edge_index.numpy().T, edge_weight = edge_index.numpy().T,
                                        features = feat_mats, targets = target_mats)
    
    return dataset,edge_weights

# Define GNN
class GCN(torch.nn.Module):
    def __init__(self,node_features):
        super(GCN,self).__init__()
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
        
        # Might need normalization here since we concatenate a value between 0 and 1
        x1 = self.fc1(x2)
        return x1
          
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
        self.fc1 = torch.nn.Linear(4, 32)      
        self.fc2 = torch.nn.Linear(32, 1)  
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)

        # Might need normalization here since we concatenate a value between 0 and 1
        x1 = torch.sum(x1, dim = 1, keepdim = True)
        x2 = torch.sum(x2, dim = 1, keepdim = True)
        x3 = torch.sum(x3, dim = 1, keepdim = True)
        
        c = F.sigmoid(x3) #Need class weights
        h = torch.cat((x1,x2,x3), dim = 1)
        x = torch.cat([h,c],dim = 1)
        
        r = self.fc1(x)
        r = r.relu()
        r = F.dropout(r, p=0.5, training = self.training)
        r = self.fc2(r)
        return r,x3

class GAT(torch.nn.Module):
    def __init__(self,node_features,heads):
        super(GAT,self).__init__()
        self.conv1 = GATConv(node_features, 32,heads, dropout=0.4)
        self.conv2 = GATConv(32*heads, 16, heads,concat=False)
        self.fc1 = torch.nn.Linear(16*heads, 1)
        self.fc2 = torch.nn.Linear(33, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        x = F.dropout(x, training=self.training)
        x2 = self.conv2(x, edge_index)
        # Might need normalization here since we concatenate a value between 0 and 1
        x1 = self.fc1(x2)
        c = F.sigmoid(x1) #Need class weights
        x = torch.cat([x,c],dim = 1)
        r = self.fc2(x)
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
        elif validation_loss > (self.min_val_loss + self.min_delta):
            self.counter +=1
            if self.counter >= self.patience:
                return True
        return False
            
# Train GNN
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
              mask = snapshot.y[:,1].detach().numpy() == 1 #Need to create a mask for nodes to train on
              cost = cost + torch.mean((y_hat[mask].squeeze()-snapshot.y[mask,0].to(device))**2)
            else:
              cost = cost + F.binary_cross_entropy_with_logits(y_hat.squeeze(), snapshot.y[:,1],pos_weight=(weight_pos))

        print(time)
        cost = cost / (time+1)
        print(cost.item())
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        val_cost = torch.tensor(0).to(device)
        if epoch <3:
            continue
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
                # Update with best weights and stop training
                count = 0
                best_weights = early_stopper.weights
                for param in model.parameters():
                    param.data = best_weights[count]
                    count +=1
                print("Early stop, best Val_Loss: ", early_stopper.min_val_loss)
                return best_weights
    return early_stopper.weights
        
# Test GNN
def eval_F1_MAE(edge_weights_Te):
    model_r.eval()
    model_c.eval()
    cost_1 = torch.tensor(0).to(device)
    cost = torch.tensor(0).to(device)
    CF = 0
    count = 0
    cost_median = 0
    with torch.no_grad():
        for time, snapshot in enumerate(test_dataset): 
            EW = edge_weights_Te[time]
            y_hat = model_c(snapshot,EW)
            pred = F.sigmoid(y_hat)
            CF = CF + confusion_matrix(snapshot.y[:,1].detach().numpy(), np.round((pred.squeeze().detach().numpy())))
            cost_1 = cost_1 + f1_score(snapshot.y[:,1].detach().numpy(), np.round((pred.squeeze().detach().numpy())), average = 'macro')
            correct_idx = (snapshot.y[:,1].detach().numpy() == 1) # Does become prevalent in a given country
            correct_idx2 = (snapshot.y[:,0].detach().numpy() != 0) # Is not already prevlanet in a given country 
            correct_idx = np.bitwise_and(correct_idx,correct_idx2)

            #y_hat has to be > and a multiple of 14
            if np.any(correct_idx):
                ## NEED TO PUT ONE MORE HERE TO SEE WHICH NODES TO PREDICT FOR
                count += 1
                y_hat = model_r(snapshot,EW)
                pred = y_hat.squeeze()[correct_idx]
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
        return CF,cost_1,cost,cost_median,pred,correct_idx

# Define the append_to_csv function
def append_to_csv(filepath, values, header=None):
    # Check if file exists
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # If the file doesn't exist, write the header first
        if not file_exists and header is not None:
            writer.writerow(header)
        
        writer.writerow(values)

T = 4

# Generate the current timestamp for the entire run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
header = ["CF1", "f11", "MAE1", "MAE2", "pred", "date", "countries"]

ITERATION_NAME = "T4"

PARENT_FOLDER = "Results"
SUB_FOLDER = f"{ITERATION_NAME}_{timestamp}"

ERROR_FILE = 'status.csv'

#Get a list of variants
variant_names = data_GT['pangoLineage'].unique().tolist()
print(variant_names)

IS_DEBUG = False

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

    # Loop over these 2 dates (every 2 weeks):
    # For each date d, train & validate on the interpolated data up till d-1
    
    start_weights_r = []
    start_weights_c = []

    for d in dates: # retrospective dates
        try:

            df = time_data[time_data['date']< (d - pd.Timedelta(days=T-1))] #Retro TRAINING data
            #One timestep or all timesteps for testing? && PER VARIANT
            
            df_GT_Te = data_GT[(data_GT['pangoLineage'] == variant)] #Retro TESTING data (data_GT['date'] == d) & 
            
            if len(df_GT_Te) ==0:
                continue
                
            dataset,edgeWeights = process_data(df,T) 
            
            #validate on approx last month of data
            100/dataset.snapshot_count
            train_dataset, val_dataset = temporal_signal_split(dataset, train_ratio=0.8)
            LL = len(train_dataset.targets)
            train_edges = edgeWeights[:LL]
            val_edges = edgeWeights[LL:]
            
            test_dataset, edgeWeightsTE = process_data_test(df_GT_Te,T,d)
            #Find class weights
            all_labels = np.concatenate(train_dataset.targets)
            classes, classes_counts = np.unique(all_labels[:,1], return_counts = True)

            weight_class = torch.tensor(len(all_labels)/(len(classes)*classes_counts))
            
            weight_pos = weight_class[1]
            
            torch.cuda.empty_cache()
            epochs = 100
            device = 'cpu'
            model_r = GCN(node_features = T + 1).to(device)
            model_c = GCN(node_features = T + 1).to(device)
            
            if len(start_weights_r) != 0:              
              param_iter = iter(model_r.parameters())
              for weight_tensor in start_weights_r:
                  param = next(param_iter)
                  with torch.no_grad():
                      param.copy_(weight_tensor)

            optimizer = torch.optim.Adam(model_r.parameters(), lr=0.05)
            early_stopper = EarlyStopper(patience=3,min_delta=5)
            model = model_r
            start_weights_r = train(T,epochs,optimizer,early_stopper, weight_pos,train_edges,val_edges, 1)
            model_r = model
            
            if len(start_weights_c) != 0:              
              param_iter = iter(model_c.parameters())
              for weight_tensor in start_weights_c:
                  param = next(param_iter)
                  with torch.no_grad():
                      param.copy_(weight_tensor)

            optimizer = torch.optim.Adam(model_c.parameters(), lr=0.01)
            early_stopper = EarlyStopper(patience=3,min_delta=0.05)
            model = model_c
            start_weights_c = train(T,epochs,optimizer,early_stopper, weight_pos,train_edges,val_edges, 0)
            model_c = model

            # Evaluate on date d USING GT:
            # Classifier f1 evaluation
            # For correctly classified 1s, find MAE
            # save thes values in lists for later plotting

            CF1, f11, MAE1, MAE2, pred, country_mask = eval_F1_MAE(edgeWeightsTE)

            # Convert 1s and 0s to True and False
            bool_country = [bool(val) for val in country_mask]

            selected_countries = list(compress(countries, bool_country))
            append_to_csv(filepath, [CF1, f11, MAE1, MAE2, pred.tolist(), d, selected_countries], header=header)
            
            print(pred)

            if MAE1 == -1:
                break
    
        except Exception as e:
            print(f"Error encountered for variant {variant}, Skipping.")
            append_to_csv(status_filepath, ['ERROR', variant, f'{e} with dates {d}'])
            continue
    
    append_to_csv(status_filepath, ['SUCCESS', variant, 'None'])

    if IS_DEBUG:
        break