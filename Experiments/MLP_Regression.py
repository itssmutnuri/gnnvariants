import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import os
from datetime import date, timedelta, datetime
import csv
import json
import warnings
from sklearn.neural_network import MLPRegressor
import torch
# from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool, global_mean_pool, GATConv, norm
import torch.nn.functional as F
# import tqdm
pre = True
# Hide all warnings
warnings.filterwarnings("ignore")
#%%
# Load the CSV data into a pandas DataFrame
data = pd.read_csv(r'..\data\NEW_DATA_RetroS.csv')

# Sort the data by 'date'
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date')

# Create a binary label column
#data['dominant'] = 1

# Create a list of unique countries
with open(r"..\data\countries_clustered.csv", mode = 'r') as file:
    reader = csv.reader(file)
    countries = []
    for row in reader:
        countries.append(row[0])

with open(r"..\data\all_vars21_clustered_NEW.csv", mode = 'r') as file:
    reader = csv.reader(file)
    all_variants = []
    for row in reader:
        all_variants.append(row[0])

all_variants = all_variants[1:]
countries = countries[1:]
S = pd.read_csv(r"..\data\growth_rates_clustered_new.csv")
# Inject artificial data for countries where it doesn't become prevalent
# for pango in all_variants:
#     data2 = data[data['pangoLineage']==pango]
#     for country in countries:
#         if country not in data2['country'].unique():
#             for date in data2['date'].unique():
#                 data = pd.concat([data,pd.DataFrame({'pangoLineage': pango,
#                                     'country': [country],
#                                     'date': [date],
#                                     'count': 0,
#                                     'prev': 0,
#                                     'days_to_prev': 0,
#                                     'pangoLineage2': [pango],
#                                     'Day': 0,
#                                     'B': 0})])
#%%
adj_mat = np.zeros((len(countries),len(countries)))


with open('..\data/country_adj_fullname.json', 'r') as json_file:
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
        
      for c in C:
          adj_mat[i, np.isin(countries,c)] = 1
#%%
def process_data(df,T):
    #df['date'] = pd.to_datetime(df['date'])
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
            #si = S.S[p_index]
            # Create the feat_matrix and target_matrix
            feat_matrix = np.zeros((len(countries), (T*2)))
            # feat_matrix = np.log(np.zeros((len(countries), (T*2))) + (10**-10))
            #feat_matrix[:,(T*2)] = 0
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
            prev_values = df[(df['date'] >= (pd.Timestamp(d) - pd.DateOffset(weeks = (T*2)-1))) &
                             (df['date'] <= d) &
                             (df['pangoLineage'] == pangoLineage)]
            countries_pres = prev_values['country'].unique()
            # what about countries which eventually get filtered out
            if len(countries_pres)==0:
                continue
            for c in countries_pres:
                temp_c = prev_values[(prev_values['country'] == c)].sort_values(by='date')
                two_week_intervals = pd.date_range(end=pd.Timestamp(d), periods=T, freq='2W-MON')
                temp_c = temp_c[temp_c['date'].isin(two_week_intervals)]
                prev_values_c = temp_c['prevsO'].values
                #print(d)
                #print(temp_c)
                row_index = countries.index(c)
                indices = np.where(adj_mat[row_index,:] == 1)[0]
                c_list_adj = [countries[i] for i in indices]
                temp_cAd = prev_values[(prev_values['country'].isin(c_list_adj))].sort_values(by='date')
                temp_cAd = temp_cAd[temp_cAd['date'].isin(two_week_intervals)]
                prev_values_cAd = (temp_cAd.groupby('date')['prevsO'].mean()).values
                
                #print(si)
                #print(temp_c)
                
                #print("???????????????????????????????????????????")
                #print(prev_values_c)
                # If no prev values were found, fill the row with 0s
                try:
                    si = temp_c['S'].values[-1]
                except:
                    si = 0
                if (len(prev_values_c) == 0) or (np.all(prev_values_c==0)):
                  prev_values_c = np.zeros((T))
                  #si = 0                  
                elif len(prev_values_c) < T:
                  prev_values_c = np.pad(prev_values_c, ((T)-len(prev_values_c), 0), 'constant')
                  si = temp_c['S'].values[-1]
                else:
                  si = temp_c['S'].values[-1]
                
                if (len(prev_values_cAd) == 0) or (np.all(prev_values_cAd==0)):
                  prev_values_cAd = np.zeros((T))
                  si = 0
                elif len(prev_values_cAd) < T:
                  prev_values_cAd = np.pad(prev_values_cAd, ((T)-len(prev_values_cAd), 0), 'constant')
                #   si = temp_c['S'].values[-1]
                # else:
                #   si = temp_c['S'].values[-1]

                log_prev_vals = prev_values_c#np.log(prev_values_c + (10**-10))
                log_prev_valsAd = prev_values_cAd#np.log(prev_values_cAd + (10**-10))
                log_prev_vals = np.append(log_prev_vals,log_prev_valsAd)
                appended_prev_si =log_prev_vals# np.append(log_prev_vals, si)
                
                row_index = countries.index(c)
                #print(feat_matrix)
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
                
                
            # snap = Data(x = feat_matrix, edge_index = edge_index, y = target_matrix)
            #print(feat_mats)
            feat_mats.append(feat_matrix)# = np.append(feat_mats,feat_matrix, axis = 0)
            target_mats.append(target_matrix) # = np.append(target_mats, target_matrix, axis = 0)
    feat_mats = np.vstack(feat_mats)
    target_mats = np.vstack(target_mats)
            
    return feat_mats, target_mats
    # batchesF.append(feat_mats)
    # batchesT.append(target_mats)
    # feat_mats=[]
    # target_mats=[]
        # torch.save(snap, f'snap_data/snap_{count}.pt')
        # count = count + 1
        
def process_data_test(df,T,d):
    #df['date'] = pd.to_datetime(df['date'])
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
        # feat_matrix = np.log(np.zeros((len(countries), (T*2))) + (10**-10))
        feat_matrix =np.zeros((len(countries), (T*2)))
        #feat_matrix[:,(T*2)] = 0
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
        prev_values = df[(df['date'] >= (pd.Timestamp(d) - pd.DateOffset(weeks = (T*2)-1))) &
                         (df['date'] <= d) &
                         (df['pangoLineage'] == pangoLineage)]
        countries_pres = prev_values['country'].unique()
        # what about countries which eventually get filtered out
        
        if len(countries_pres)==0:
            continue
        for c in countries_pres:
            temp_c = prev_values[(prev_values['country'] == c)].sort_values(by='date')
            two_week_intervals = pd.date_range(end=pd.Timestamp(d), periods=T, freq='2W-MON')
            temp_c = temp_c[temp_c['date'].isin(two_week_intervals)]
            prev_values_c = temp_c['prevsO'].values
            if all(temp_c['prev'].values == 0): #Fixes some noisy data during evaluation
                prev_values_c = temp_c['prev'].values
            #print(d)
            #print(temp_c)
            row_index = countries.index(c)
            indices = np.where(adj_mat[row_index,:] == 1)[0]
            c_list_adj = [countries[i] for i in indices]
            temp_cAd = prev_values[(prev_values['country'].isin(c_list_adj))].sort_values(by='date')
            temp_cAd = temp_cAd[temp_cAd['date'].isin(two_week_intervals)]
            prev_values_cAd = (temp_cAd.groupby('date')['prevsO'].mean()).values
            
            #print(si)
            #print(temp_c)
            
            #print("???????????????????????????????????????????")
            #print(prev_values_c)
            # If no prev values were found, fill the row with 0s
            try:
                si = temp_c['S'].values[-1]
            except:
                si = 0
            if (len(prev_values_c) == 0) or (np.all(prev_values_c==0)):
              prev_values_c = np.zeros((T))
              #si = 0                  
            elif len(prev_values_c) < T:
              prev_values_c = np.pad(prev_values_c, ((T)-len(prev_values_c), 0), 'constant')
              si = temp_c['S'].values[-1]
            else:
              si = temp_c['S'].values[-1]
            
            if (len(prev_values_cAd) == 0) or (np.all(prev_values_cAd==0)):
              prev_values_cAd = np.zeros((T))
              si = 0
            elif len(prev_values_cAd) < T:
              prev_values_cAd = np.pad(prev_values_cAd, ((T)-len(prev_values_cAd), 0), 'constant')
            #   si = temp_c['S'].values[-1]
            # else:
            #   si = temp_c['S'].values[-1]

            log_prev_vals = prev_values_c#np.log(prev_values_c + (10**-10))
            log_prev_valsAd = prev_values_cAd#np.log(prev_values_cAd + (10**-10))
            log_prev_vals = np.append(log_prev_vals,log_prev_valsAd)
            appended_prev_si = log_prev_vals#np.append(log_prev_vals, si)
            
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
            
            
        # snap = Data(x = feat_matrix, edge_index = edge_index, y = target_matrix)
        
        feat_mats.append(feat_matrix)# = np.append(feat_mats,feat_matrix, axis = 0)
        target_mats.append(target_matrix) # = np.append(target_mats, target_matrix, axis = 0)
    feat_mats = np.vstack(feat_mats)
    target_mats = np.vstack(target_mats)
    return  feat_mats, target_mats
#%%
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

#%%
# Prepare the features and labels
T = 4

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
header = ["CF1", "f11", "MAE1", "MAE2", "pred", "date", "countries"]

ITERATION_NAME = "MLP_R_FIXED_prev_NoS"

PARENT_FOLDER = "Results"
SUB_FOLDER = f"{ITERATION_NAME}_{timestamp}"

ERROR_FILE = 'status.csv'
IS_DEBUG = False
for pango in all_variants:
    print(f" ======================== RUNNING FOR {pango} ======================== ")
    data['date'] = pd.to_datetime(data['date'])
    data2 = data[data['pangoLineage']==pango]
    dates = data2['date'].unique()
    dates = pd.to_datetime(dates).tolist()
    dates.sort()
    directory = f"{PARENT_FOLDER}/{SUB_FOLDER}"
    filename = f"{pango}.csv"
    filepath = os.path.join(directory, filename)
    status_filepath = os.path.join(directory, ERROR_FILE)
    
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for d in dates:
        if d == dates[0] and pango==all_variants[0]:
            continue
        if (d == dates[0] or d == dates[1]) and pango==all_variants[1]:
            continue
        df = data[data['date']< (d - pd.Timedelta(days=T-1))] #Retro TRAINING data
        # print(df)
            # df_GT = data_GT[data_GT['date']<=d]
            #One timestep or all timesteps for testing? && PER VARIANT
        df_GT_Te = data2 #Retro TESTING data (data_GT['date'] == d) & 
        
        if len(df_GT_Te) ==0 or len(df) == 0:
            continue
            
        
        
        X_train, y_train = process_data(df,T)
        y_train = y_train[:,0]
        # for idx in range(T, len(df)):
        #     prev_values = data['prev'].iloc[idx - T:idx].values
        #     X_train.append(prev_values)
        #     y_train.append(data['B'].iloc[idx])
        
        
        # X_test = []
        # y_test = []
        # for idx in range(T, len(df)):
        #     prev_values = data['prev'].iloc[idx - T:idx].values
        #     X_test.append(prev_values)
        #     y_test.append(data['B'].iloc[idx])
        # Split the data into training and testing sets
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_test, y_test  = process_data_test(df_GT_Te,T,d)
        
        correct_idx = (y_test[:,1] == 1) #Does become prevalent in a given country
        correct_idx2 = (y_test[:,0] != 0) #Is not already prevlanet in a given country 
        correct_idx = np.bitwise_and(correct_idx,correct_idx2)
        
        
        # correct_idx = np.bitwise_and(correct_idx,snapshot.y[:,0].detach().numpy() !=0)
        #y_hat has to be > and a multiple of 14
        count = 0
        mask = np.ones_like(y_test[:,1], dtype=bool)
        if pre:
            mask = np.all(X_test[:,0:T] == np.log(0 + (10**-10)),axis=1)
        # correct_idx = np.bitwise_and(correct_idx,snapshot.y[:,0].detach().numpy() !=0)
        #y_hat has to be > and a multiple of 14
        if not (np.any(np.bitwise_and(correct_idx,mask))):
        #if sum(y_test[:,0]) == 0: #It became prevalent everywhere
            append_to_csv(filepath, [0, 0, -1, -1, [], d, []], header=header)
            break
        y_test = y_test[:,0]
        #print("OK")
        #Find class weights
        # all_labels = np.concatenate(y_train)
        # classes, classes_counts = np.unique(all_labels[:,1], return_counts = True)

        # weight_class = len(all_labels)/(len(classes)*classes_counts)
        
        # weight_pos = weight_class[1]
        # Train the Decision Tree model
        clf = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=100, activation='relu', solver='adam', random_state=42)
        clf.fit(X_train, y_train)
        
        # Make predictions on the testing set
        # y_pred = clf.predict(X_test)
        
        CF = 0
        cost_1 = 0
        cost = 0
        cost_median = 0
        # count += 1
        y_hat = clf.predict(X_test)
        if pre:
            pred = y_hat[correct_idx & mask]
            cost = cost + np.mean(np.abs(((np.ceil(np.maximum(pred,1)/14)*14)-y_test[correct_idx & mask])))
            cost_median = cost_median + np.median(np.abs(((np.ceil(np.maximum(pred,1)/14)*14)-y_test[correct_idx & mask])))
            append_to_csv(filepath, [0, 0, cost, cost_median, [], d, []], header=header)
        else:
            pred = y_hat[correct_idx]
            cost = cost + np.mean(np.abs(((np.ceil(np.maximum(pred,1)/14)*14)-y_test[correct_idx])))
            cost_median = cost_median + np.median(np.abs(((np.ceil(np.maximum(pred,1)/14)*14)-y_test[correct_idx])))
            append_to_csv(filepath, [0, 0, cost, cost_median, [], d, []], header=header)
        # if count !=0:           
        #     cost = cost / (count)
        #     cost_median = cost_median/ count       
        #     cost = cost.item()
        #     cost_median =  cost_median.item()
        # else:
        #     cost = -1
        #     cost_median = -1
        # CF = CF + confusion_matrix(y_test, np.round((y_pred)))
        # cost_1 = cost_1 + f1_score(y_test, np.round((y_pred)), average = 'macro')
        append_to_csv(filepath, [0, 0, cost, cost_median, [], d, []], header=header)
        # Evaluate the model
        #accuracy = accuracy_score(y_test, y_pred)
        print(f"MLP_R Loss: {cost_median}")
    append_to_csv(status_filepath, ['SUCCESS', pango, 'None'])
