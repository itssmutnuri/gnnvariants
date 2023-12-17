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

# %% Test

nb_of_countries = 0
time_data = pd.read_csv(r"NEW_DATA_RetroS.csv")
# Load GT
data_GT = pd.read_csv(r"NEW_DATA_RetroS.csv")  # Processed_res21_1_clusteredGT.csv")
restrictions = pd.read_csv(r"international_travel_covid.csv")
# list(set(variants['country']))
with open(r"countries_clustered.csv", mode='r') as file:
    reader = csv.reader(file)
    countries = []
    for row in reader:
        countries.append(row[0])

with open(r"all_vars21_clustered_NEW.csv", mode='r') as file:
    reader = csv.reader(file)
    all_variants = []
    for row in reader:
        all_variants.append(row[0])

all_variants = all_variants[1:]
print("\n \n \n Replaced....")

countires_NA = ["Union of the Comoros", "Liechtenstein", "Kosovo", "Timor-Leste"]
countries = countries[1:]

if nb_of_countries != 0:
    filtered_country_names = []
    top_X_countries = time_data['country'].value_counts().nlargest(nb_of_countries).index.tolist()
    # print(top_X_countries)
    # Now, you have a list of the top 60 countries with the most data
    # You can use this list to filter your DataFrame, for example:
    time_data = time_data[time_data['country'].isin(top_X_countries)]
    data_GT = data_GT[data_GT['country'].isin(top_X_countries)]
    for country in countries:
        if country in top_X_countries:
            filtered_country_names.append(country)

    countries = filtered_country_names


# loader = ChickenpoxDatasetLoader()

# dataset = loader.get_dataset()

# train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# for time, snapshot in enumerate(train_dataset):
#     x = snapshot.x
#     edge_index = snapshot.edge_index
#     edge_attr = snapshot.edge_attr
#         # y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
#         # cost = cost + torch.mean((y_hat-snapshot.y)**2)
# %%
def checkCountries(c1, c2):
    for c in c1:
        if c in c2:
            continue
        else:
            print(c, difflib.get_close_matches(c, c2))


# %% Define adjacency matrix
routes = pd.read_csv(r"data/routes.dat", header=None)
airports = pd.read_csv(r"data/airports.dat", header=None)  # 4th is country

# Country-AirportID
IATA_country = np.transpose(np.array([list(airports[3]), list(airports[0])]))
# ICAO_country = np.transpose(np.array([list(airports[3]), list(airports[5])]))

rou = np.transpose(np.array([list(routes[3]), list(routes[5])]))
route = np.empty(rou.shape, dtype=np.dtype('U100'))
for i in range(len(rou)):
    # if (rou[i][0]=='\\N') or (rou[i][1]=='\\N'):
    #     rou = np.delete(rou,i,0)
    #     route = np.delete(route,i,0)
    try:
        while (np.where(IATA_country[:, 1] == rou[i][1])[0].size == 0) or (
                np.where(IATA_country[:, 1] == rou[i][0])[0].size == 0) or (rou[i][0] == '\\N') or (rou[i][1] == '\\N'):
            rou = np.delete(rou, i, 0)
            route = np.delete(route, i, 0)
    except:
        break

    route[i][0] = str(IATA_country[np.where(IATA_country[:, 1] == rou[i][0]), 0][0][0])
    route[i][1] = str(IATA_country[np.where(IATA_country[:, 1] == rou[i][1]), 0][0][0])

variants = pd.read_csv(r"data/global_vars_May22.csv")  # 3rd

# countries = pd.read_csv(r"countries.csv")
# all_vars = pd.read_csv(r"all_vars21.csv")

# Make sure countries are spelt the same between variants and airport dataset
# checkCountries(countries,IATA_country[:,0])

# United States->USA
route[:, 0] = np.char.replace(route[:, 0], "United States", "USA")
route[:, 1] = np.char.replace(route[:, 1], "United States", "USA")

# list(set(variants['country']))

adj_mat = np.eye(len(countries))

# Aruba are islands and go to the else since they have no adj

with open('data/country_adj_fullname.json', 'r') as json_file:
    data_json = json.load(json_file)
    for i in range(len(countries)):
        # print(data_json)
        # print(countries[i])
        # try:
        if countries[i] == "Aruba" or countries[i] == "Fiji" or countries[i] == "Guadeloupe" or countries[
            i] == "Iceland" or countries[i] == "Jamaica" or countries[i] == "Maldives" or countries[i] == "Mauritius" or \
                countries[i] == "New Zealand" or countries[i] == "Seychelles":
            C = []
        elif countries[i] == "Hong Kong":
            C = ["China"]
        elif countries[i] == "Russia":
            C = data_json["Russian Federation"]
        elif countries[i] == "USA":
            C = data_json["United States of America"]
        else:
            C = data_json[countries[i]]
        # except Exception as e:
        # print("No adjacency:", e)
        # C = []
        # print(C)
        # source_table = route[route[:,0] ==countries[i]]
        # print(source_table)
        # adj_mat[i, np.isin(countries,source_table[:,1])] = 1
        for c in C:
            adj_mat[i, np.isin(countries, c)] = 1

# np.count_nonzero(adj_mat,1) #~20 no routes

# %%
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


# %% Load S and cap if needed
def remove_outliers(data):
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)].dropna()
    return clean_data


S = pd.read_csv(r"growth_rates_clustered_new.csv")


##Normalize data with S
# %% Define time-series data matrix
# df = time_data
# # Define the number of days to go back in the snapshot matrices
# T = 14

# # 0: never reaches prev, 1: hasn't reached prev, 2: reached prev
# temp_retro = "21J.Delta"
def edgeW_calc(df):
    weighted_mat = np.ones((len(countries), len(countries)))

    # sort by restriction lvls 0->4
    df = df.sort_values(by='international_travel_controls')

    for c in df['country'].unique():
        original_value = df[df['country'] == c]['international_travel_controls'].values[0]
        scaled_value = 1 - (original_value / 4) * 0.9
        indx = np.where(countries == c)
        weighted_mat[indx, :] = scaled_value
        weighted_mat[:, indx] = scaled_value

    # Mask the unnecessary ones
    weighted_mat = np.where(adj_mat == 0, 0, weighted_mat)
    for i in range(len(countries)):
        weighted_mat[i, i] = 1

    # print(weighted_mat)
    weight_tensor = torch.DoubleTensor(weighted_mat)

    edge_weights = []
    for e in range(len(edge_index)):
        edge_idx = edge_index[e]
        i = edge_idx[0]
        j = edge_idx[1]
        edge_weights.append(weight_tensor[i][j])

    # print(edge_weights.shape)
    edge_weights = torch.DoubleTensor(edge_weights)

    return edge_weights


# Get the unique dates and pangoLineages in the DataFrame
def process_data(df, T):
    # df['date'] = pd.to_datetime(df['date'])
    dates = df['date'].unique()  # df['date'].unique()
    # pangoLineages = all_variants
    # df['pangoLineage'].unique()
    # date_0 = "2020-04-28";
    feat_mats = []  # np.empty((0,len(countries),T+1))
    target_mats = []  # np.array((0,len(countries),2))
    edge_weights = []
    # Loop over each date and pangoLineage (each gives us a snapshot)
    # count = 0
    # batches = []
    # batchesT = []
    for d in dates:
        pangos = df[(df['date'] == d)]
        pangos = pangos['pangoLineage'].unique()
        # print(restrictions)
        controls = restrictions[restrictions['date'] == d]

        EW = edgeW_calc(controls)
        # print(controls)
        # controls_array = np.array([controls[controls['country']==c]['international_travel_controls'].values[0] for c in countries])
        # print(controls_array)
        # what if we have a ariant not present at a given dat, but is at other dates?
        # batch each varaint group  and dont do this maybe?
        for pangoLineage in pangos:  # pangoLineages:
            p_index = all_variants.index(pangoLineage)
            # si = S.S[p_index]
            # Create the feat_matrix and target_matrix
            feat_matrix = np.zeros((len(countries), T))
            target_matrix = np.zeros((len(countries), 2))
            target_matrix[:, 0] = -1  # Will never reach prevalence
            countries_dom = df[(df['pangoLineage'] == pangoLineage) & (df['prev'] > 1 / 3)]

            k1 = countries_dom.drop_duplicates(subset='country', keep='first')
            idx2 = (k1.date - d).dt.days
            idx2[idx2 < 0] = 0
            # Check if date of dom has passed, if it did then 0, else calculate it.
            idx = [countries.index(si) for si in countries_dom['country'].unique()]
            target_matrix[idx, 0] = idx2
            target_matrix[idx, 1] = 1
            # Get the prev values for the current date and pangoLineage
            # prev_values = df[(pd.to_datetime(df['date']) >= pd.to_datetime(d) - pd.Timedelta(days=T-1)) &
            #                  (pd.to_datetime(df['date']) <= pd.to_datetime(d)) &
            #                  (df['pangoLineage'] == pangoLineage)]

            # prev_values = df[(df['Day'] >= (d) - (T-1)) &
            #                  (df['Day'] <= d) &
            #                  (df['pangoLineage'] == pangoLineage)]
            prev_values = df[(df['date'] >= (pd.Timestamp(d) - pd.DateOffset(weeks=(T * 2) - 1))) &
                             (df['date'] <= d) &
                             (df['pangoLineage'] == pangoLineage)]
            countries_pres = prev_values['country'].unique()
            # what about countries which eventually get filtered out
            if len(countries_pres) == 0:
                continue
            for c in countries_pres:
                temp_c = prev_values[(prev_values['country'] == c)].sort_values(by='date')
                # print(d)
                two_week_intervals = pd.date_range(end=pd.Timestamp(d), periods=T, freq='2W-MON')
                # print(two_week_intervals)
                # print(temp_c)
                temp_c = temp_c[temp_c['date'].isin(two_week_intervals)]
                # print(temp_c)
                prev_values_c = temp_c['prevsO'].values
                # print(d)
                # print(temp_c)

                # print(si)
                # prev_values_c = prev_values[(prev_values['country'] == c)]['prev'].values
                # If no prev values were found, fill the row with 0s
                if len(prev_values_c) == 0:
                    prev_values_c = np.zeros(T)
                    si = 0
                # If not enough prev values were found, pad with 0s
                elif len(prev_values_c) < T:
                    prev_values_c = np.pad(prev_values_c, (T - len(prev_values_c), 0), 'constant')
                    si = temp_c['S'].values[-1]
                else:
                    si = temp_c['S'].values[-1]

                log_prev_vals = np.log(prev_values_c + (10 ** -10))
                appended_prev_si = prev_values_c#np.append(log_prev_vals, si)
                row_index = countries.index(c)

                feat_matrix[row_index, :] = appended_prev_si

                # Get the days_to_prev value for the current date and pangoLineage
                target_vals = prev_values[(prev_values['date'] == d) & (prev_values['country'] == c)]

                days_to_prev = target_vals['days_to_prev'].values
                dom = target_vals['B'].values
                # If no days_to_prev value was found, set it to 10000 (doesn't become prevalent)
                if len(days_to_prev) == 0:
                    continue

                # Set the value in the target_matrix to the days_to_prev value
                target_matrix[row_index, 0] = days_to_prev
                target_matrix[row_index, 1] = dom

            # feat_matrix = np.hstack((feat_matrix, controls_array.reshape(-1, 1)))
            # snap = Data(x = feat_matrix, edge_index = edge_index, y = target_matrix)
            feat_mats.append(feat_matrix)  # = np.append(feat_mats,feat_matrix, axis = 0)
            target_mats.append(target_matrix)  # = np.append(target_mats, target_matrix, axis = 0)
            edge_weights.append(EW)
    dataset = StaticGraphTemporalSignal(edge_index=edge_index.numpy().T, edge_weight=edge_index.numpy().T,
                                        features=feat_mats, targets=target_mats)

    return dataset, edge_weights
    # batchesF.append(feat_mats)
    # batchesT.append(target_mats)
    # feat_mats=[]
    # target_mats=[]
    # torch.save(snap, f'snap_data/snap_{count}.pt')
    # count = count + 1


def process_data_test(df, T, d):
    # df['date'] = pd.to_datetime(df['date'])
    # dates = df['date'].unique()# df['date'].unique()
    # pangoLineages = all_variants
    # df['pangoLineage'].unique()
    # date_0 = "2020-04-28";

    # Here get the passed 14 days of data (interpolated features)

    feat_mats = []  # np.empty((0,len(countries),T+1))
    target_mats = []  # np.array((0,len(countries),2))
    # Loop over each date and pangoLineage (each gives us a snapshot)
    edge_weights = []
    # count = 0
    # batches = []
    # batchesT = []
    # pangos = df[(df['date'] == d)]
    pangos = df['pangoLineage'].unique()
    controls = restrictions[restrictions['date'] == d]
    EW = edgeW_calc(controls)
    # controls_array = np.array([controls[controls['country']==c]['international_travel_controls'].values[0] for c in countries])
    # what if we have a ariant not present at a given dat, but is at other dates?
    # batch each varaint group  and dont do this maybe?
    for pangoLineage in pangos:  # pangoLineages:
        p_index = all_variants.index(pangoLineage)
        si = S.S[p_index]
        # Create the feat_matrix and target_matrix
        feat_matrix = np.zeros((len(countries), T))
        target_matrix = np.zeros((len(countries), 2))
        target_matrix[:, 0] = -1  # Will never reach prevalence
        countries_dom = df[(df['pangoLineage'] == pangoLineage) & (df['prev'] > 1 / 3)]

        k1 = countries_dom.drop_duplicates(subset='country', keep='first')
        idx2 = (k1.date - d).dt.days
        idx2[idx2 < 0] = 0
        # Check if date of dom has passed, if it did then 0, else calculate it.
        idx = [countries.index(si) for si in countries_dom['country'].unique()]
        target_matrix[idx, 0] = idx2
        target_matrix[idx, 1] = 1
        # Get the prev values for the current date and pangoLineage
        # prev_values = df[(pd.to_datetime(df['date']) >= pd.to_datetime(d) - pd.Timedelta(days=T-1)) &
        #                  (pd.to_datetime(df['date']) <= pd.to_datetime(d)) &
        #                  (df['pangoLineage'] == pangoLineage)]

        # prev_values = df[(df['Day'] >= (d) - (T-1)) &
        #                  (df['Day'] <= d) &
        #                  (df['pangoLineage'] == pangoLineage)]
        prev_values = df[(df['date'] >= (pd.Timestamp(d) - pd.DateOffset(weeks=(T * 2) - 1))) &
                         (df['date'] <= d) &
                         (df['pangoLineage'] == pangoLineage)]
        countries_pres = prev_values['country'].unique()
        # what about countries which eventually get filtered out

        if len(countries_pres) == 0:
            continue
        for c in countries_pres:
            temp_c = prev_values[(prev_values['country'] == c)].sort_values(by='date')
            # print(temp_c)
            two_week_intervals = pd.date_range(end=pd.Timestamp(d), periods=T, freq='2W-MON')
            temp_c = temp_c[temp_c['date'].isin(two_week_intervals)]
            prev_values_c = temp_c['prevsO'].values
            # print(d)
            # print(temp_c)

            # print(si)
            # print(temp_c)

            # print("???????????????????????????????????????????")
            # print(prev_values_c)
            # If no prev values were found, fill the row with 0s
            if len(prev_values_c) == 0:
                prev_values_c = np.zeros(T)
                si = 0
            elif len(prev_values_c) < T:
                prev_values_c = np.pad(prev_values_c, (T - len(prev_values_c), 0), 'constant')
                si = temp_c['S'].values[-1]
            else:
                si = temp_c['S'].values[-1]

            log_prev_vals = np.log(prev_values_c + (10 ** -10))
            appended_prev_si = prev_values_c#np.append(log_prev_vals, si)
            row_index = countries.index(c)

            feat_matrix[row_index, :] = appended_prev_si

            # Get the days_to_prev value for the current date and pangoLineage
            target_vals = prev_values[(prev_values['date'] == d) & (prev_values['country'] == c)]

            days_to_prev = target_vals['days_to_prev'].values
            dom = target_vals['B'].values
            # If no days_to_prev value was found, set it to 10000 (doesn't become prevalent)
            if len(days_to_prev) == 0:
                continue

            # Set the value in the target_matrix to the days_to_prev value
            target_matrix[row_index, 0] = days_to_prev
            target_matrix[row_index, 1] = dom

        # snap = Data(x = feat_matrix, edge_index = edge_index, y = target_matrix)
        # feat_matrix = np.hstack((feat_matrix, controls_array.reshape(-1, 1)))
        feat_mats.append(feat_matrix)  # = np.append(feat_mats,feat_matrix, axis = 0)
        target_mats.append(target_matrix)  # = np.append(target_mats, target_matrix, axis = 0)
        edge_weights.append(EW)

    dataset = StaticGraphTemporalSignal(edge_index=edge_index.numpy().T, edge_weight=edge_index.numpy().T,
                                        features=feat_mats, targets=target_mats)

    return dataset, edge_weights


# %%
class GCN_r(torch.nn.Module):
    def __init__(self, node_features):
        super(GCN_r, self).__init__()
        self.conv1 = GCNConv(node_features, 32)
        self.conv2 = GCNConv(32, 16)
        # self.norm = torch.nn.LayerNorm(32)
        self.fc1 = torch.nn.Linear(16, 1)
        # self.fc2 = torch.nn.Linear(33, 1)

    def forward(self, data, edge_weight, norm=False):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)
        x = F.leaky_relu(x1)
        x = F.dropout(x, training=self.training)
        x2 = self.conv2(x, edge_index)
        x2 = F.leaky_relu(x2)
        # print(x.shape)
        # Might need normalization here since we concatenate a value between 0 and 1
        # x = self.norm(x)
        # x1 = self.conv2(x, edge_index)
        x1 = self.fc1(x2)
        # c = F.sigmoid(x1) #Need class weights
        # x = torch.cat([x,c],dim = 1)
        # print(x.shape)
        # print(c.shape)

        # r = self.fc2(x)
        # r = F.relu(r0)
        # mask = c < 0.5
        # r[mask] = -1
        # r = F.relu(r)
        return x1


class GCN_c(torch.nn.Module):
    def __init__(self, node_features):
        super(GCN_c, self).__init__()

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
        #x1 = torch.cat((x1.flatten(1), x[:, -1].unsqueeze(1)), dim=1)
        x1 = self.norm1(x1)
        x1 = F.leaky_relu(x1)

        # x2 = self.conv1(x1, edge_index, edge_weight)
        # x2 = self.norm2(x2)
        # x2 = F.leaky_relu(x2)
        # x2 = F.dropout(x2, training=self.training, p=0.5)
        #
        # x3 = self.conv2(x2.float(), edge_index, edge_weight)
        # x3 = self.norm3(x3)
        # x3 = F.leaky_relu(x3)

        x4 = self.fc1(x1.float())
        return x4


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


# %% Train GNN
def train(T, epochs, optimizer, early_stopper, weight_pos, edge_weights_T, edge_weights_V, reg=1):
    torch.cuda.empty_cache()
    scheduler = sch.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=100)
    for epoch in tqdm(range(epochs)):
        model.train()
        cost_1 = torch.tensor(0).to(device)
        cost = torch.tensor(0).to(device)
        optimizer.zero_grad()
        for time, snapshot in enumerate(train_dataset):
            EW = edge_weights_T[time]
            # np.set_printoptions(threshold=sys.maxsize)
            # print(EW)
            y_hat = model(snapshot, EW)  # .x.to(device), snapshot.edge_index.to(device), snapshot.edge_attr.to(device))

            if reg:
                y_hat = F.relu(y_hat)
                mask = snapshot.y[:, 1].detach().numpy() == 1  # Need to create a mask for nodes to train on
                cost = cost + torch.mean((y_hat[mask].squeeze() - snapshot.y[mask, 0].to(device)) ** 2)
            else:
                cost = cost + F.binary_cross_entropy_with_logits(y_hat.squeeze(), snapshot.y[:, 1],
                                                                 pos_weight=(weight_pos))

            # cost = cost + cost_2 + cost_1

        print(time)
        cost = cost / (time + 1)
        print(cost.item())
        cost.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        val_cost = torch.tensor(0).to(device)
        if epoch < 3:
            continue
        with torch.no_grad():
            for time, snapshot in enumerate(val_dataset):
                # val_cost = val_cost + torch.mean(torch.abs((y_hat-snapshot.y)))
                # cost_1 =  F.binary_cross_entropy(y_hat[1].squeeze(), snapshot.y[:,1],weight = weight_class)#loss_1(y_hat[1].squeeze(), snapshot.y[:,1])
                # cost_1 = F.binary_cross_entropy_with_logits(y_hat[1].squeeze(), snapshot.y[:,1],pos_weight=(weight_pos))
                # val_cost = val_cost + torch.mean((y_hat[0].squeeze()-snapshot.y[:,0].to(device))**2) + cost_1
                EW = edge_weights_V[time]
                if reg:
                    y_hat = model(snapshot, EW)
                    mask = snapshot.y[:, 1].detach().numpy() == 1
                    val_cost = val_cost + torch.mean((y_hat[mask].squeeze() - snapshot.y[mask, 0].to(device)) ** 2)
                else:
                    y_hat = model(snapshot, EW, True)
                    val_cost = val_cost + F.binary_cross_entropy_with_logits(y_hat.squeeze(), snapshot.y[:, 1],
                                                                             pos_weight=(weight_pos))

            val_cost = val_cost / (time + 1)
            val_cost = val_cost.item()
            print("Val _cost: {:.4f}".format(val_cost))
            if early_stopper.early_stop(val_cost, model.parameters()):
                # Update with best weights and stop training
                count = 0
                best_weights = early_stopper.weights
                for param in model.parameters():
                    param.data = best_weights[count]
                    count += 1
                print("Early stop, best Val_Loss: ", early_stopper.min_val_loss)
                return best_weights
    return early_stopper.weights


# %% Test GNN
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
            y_hat = model_c(snapshot, EW)
            pred = F.sigmoid(y_hat)  # F.sigmoid(y_hat[1])
            CF = CF + confusion_matrix(snapshot.y[:, 1].detach().numpy(), np.round((pred.squeeze().detach().numpy())))
            cost_1 = cost_1 + f1_score(snapshot.y[:, 1].detach().numpy(), np.round((pred.squeeze().detach().numpy())),
                                       average='macro')
            # correct_idx = np.bitwise_and(snapshot.y[:,1].detach().numpy() == np.round((pred.squeeze().detach().numpy())), (snapshot.y[:,1].detach().numpy() == 1))
            correct_idx = (snapshot.y[:, 1].detach().numpy() == 1)  # Does become prevalent in a given country
            correct_idx2 = (snapshot.y[:, 0].detach().numpy() != 0)  # Is not already prevlanet in a given country
            correct_idx = np.bitwise_and(correct_idx, correct_idx2)

            # correct_idx = np.bitwise_and(correct_idx,snapshot.y[:,0].detach().numpy() !=0)
            # y_hat has to be > and a multiple of 14
            if np.any(correct_idx):
                ## NEED TO PUT ONE MORE HERE TO SEE WHICH NODES TO PREDICT FOR
                # correct_idx = np.bitwise_and(np.round((pred.squeeze().detach().numpy())) == 1, correct_idx)
                count += 1
                y_hat = model_r(snapshot, EW)
                pred = y_hat.squeeze()[correct_idx]
                cost = cost + torch.mean(
                    torch.abs(((np.ceil(np.maximum(pred, 1) / 14) * 14) - snapshot.y[correct_idx, 0])))
                cost_median = cost_median + torch.median(
                    torch.abs(((np.ceil(np.maximum(pred, 1) / 14) * 14) - snapshot.y[correct_idx, 0])))
        cost_1 = cost_1 / (time + 1)
        cost_1 = cost_1.item()
        if count != 0:
            cost = cost / (count)
            cost_median = cost_median / count
            cost = cost.item()
            cost_median = cost_median.item()
        else:
            cost = -1
            cost_median = -1

        print("CF", CF)

        print("average F1: {:.4f}".format(cost_1))

        print("average MAE: {:.4f}".format(cost))

        print("average median AE:", cost_median)
        return CF, cost_1, cost, cost_median, pred, correct_idx


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

ITERATION_NAME = "RNN_prev_WoS"

PARENT_FOLDER = "Results"
SUB_FOLDER = f"{ITERATION_NAME}_{timestamp}"

ERROR_FILE = 'status.csv'

# Get a list of variants
variant_names = data_GT['pangoLineage'].unique().tolist()
# variant_names = variant_names[-5:]
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

    for d in dates:  # retrospective dates
        try:

            df = time_data[time_data['date'] < (d - pd.Timedelta(days=T - 1))]  # Retro TRAINING data
            # df_GT = data_GT[data_GT['date']<=d]
            # One timestep or all timesteps for testing? && PER VARIANT

            df_GT_Te = data_GT[(data_GT['pangoLineage'] == variant)]  # Retro TESTING data (data_GT['date'] == d) &

            if len(df_GT_Te) == 0:
                continue

            dataset, edgeWeights = process_data(df, T)

            # validate on approx last month of data
            100 / dataset.snapshot_count
            train_dataset, val_dataset = temporal_signal_split(dataset, train_ratio=0.8)
            LL = len(train_dataset.targets)
            train_edges = edgeWeights[:LL]
            val_edges = edgeWeights[LL:]

            test_dataset, edgeWeightsTE = process_data_test(df_GT_Te, T, d)
            # print("OK")
            # Find class weights
            all_labels = np.concatenate(train_dataset.targets)
            classes, classes_counts = np.unique(all_labels[:, 1], return_counts=True)

            weight_class = torch.tensor(len(all_labels) / (len(classes) * classes_counts))

            weight_pos = weight_class[1]

            torch.cuda.empty_cache()
            epochs = 100
            device = 'cpu'
            model_r = GCN_c(node_features=T + 1).to(device)
            model_c = GCN_c(node_features=T + 1).to(device)

            if len(start_weights_r) != 0:
              param_iter = iter(model_r.parameters())
              for weight_tensor in start_weights_r:
                  param = next(param_iter)
                  with torch.no_grad():
                      #print(weight_tensor)
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

            # print("OK6")
            optimizer = torch.optim.Adam(model_c.parameters(), lr=0.05)
            early_stopper = EarlyStopper(patience=3, min_delta=0.005)
            model = model_c
            start_weights_c = train(T, epochs, optimizer, early_stopper, weight_pos, train_edges, val_edges, 0)
            model_c = model
            # print(start_weights)

            # Evaluate on date d USING GT:
            # Classifier f1 evaluation
            # For correctly classified 1s, find MAE
            # save thes values in lists for later plotting

            # CF1,f11,MAE1,MAE2,pred = eval_F1_MAE()
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