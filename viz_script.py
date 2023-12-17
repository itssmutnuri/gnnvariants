# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:46:03 2023

@author: 16613
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import copy



#%%
# Directory containing CSV files
directory_name = "RCN_GCN_r" 
csv_directory = r"C:\USC\Research/" + directory_name

# User-defined metric
chosen_metric = 'MAE2'#'MAE1'#'f11'


# Function to calculate biweekly dates
def biweekly_dates(date_list):
    current_date = date_list.iloc[-1]

    dates = []
    dates.append(current_date)
    count = 0
    while current_date >= date_list.iloc[0] and count < 2:

        current_date -= timedelta(weeks=2)

        if not (current_date in date_list.values):

            count+=1
            continue
        else:
            count = 0 #resets if not consecutive
        dates.append(current_date)
        #print(dates)
    dates.reverse()
    return dates

# Function to calculate days till prevalence
def calculate_days_till_prevalence(date_column, metric_column):
    prevalence_date = date_column.iloc[-1]
    days_till_prevalence = []
    
    for date, metric in zip(date_column, metric_column):
        #print(date,metric)
        days_till_prevalence.append((date - prevalence_date).days)
    
    return days_till_prevalence

# Initialize dictionaries to store data
data = {}
data_2 = {}
averages = {}
medians = {}
total_values = []
total_values_median = []
date_max = []
nb_countries = {}
# Loop through all CSV files in the directory
for csv_file in os.listdir(csv_directory):
    if csv_file.endswith(".csv") and csv_file != "status.csv":
        
        variant = os.path.splitext(csv_file)[0]
        file_path = os.path.join(csv_directory, csv_file)
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        if df['MAE2'].iloc[0] == -1:
            continue
        # Calculate days till prevalence
        days_till_prevalence = calculate_days_till_prevalence(df['date'], df['MAE2'])
        
        # Store data     
        # tempp = df['date']
        index = biweekly_dates(df['date'])

        if csv_file == "21A.Delta.S.K417.csv":
            df_temp = df
            index_temp = index
        
        indices = [df.index[df['date'] == timestamp][0] for timestamp in index]
        temp = df.iloc[indices]
        temp = temp.iloc[:-1]
        
        data[variant] = temp[chosen_metric]
        temp2 = pd.Series((-1*np.array(days_till_prevalence))[indices])
        data_2[variant] = temp2[:-1]
        temp3 =  df['countries'].iloc[indices]
        temp3 = temp3.iloc[:-1]
        nb_countries[variant] = temp3.apply(eval).apply(len)
        
        if len(date_max) < len(data_2[variant]):
            date_max = data_2[variant]
        # Calculate and store the average performance over time
        average_performance = np.mean(temp[chosen_metric])
        averages[variant] = average_performance
        medians[variant] = np.median(temp[chosen_metric])
        total_values.append(average_performance)
        print(variant, average_performance)
        total_values_median.append(np.median(temp[chosen_metric]))

# find mask for equal rep:
mask = list(data.keys())[:4] + ["21H.Mu", "21G.Lambda", "21B.Kappa", "21A.Delta", "20J.Gamma.V3", "20H.Beta.V2"]
mask = np.isin(list(data.keys()), mask)  
mask = (~mask)
total_values = np.array(total_values)
total_values_median = np.array(total_values_median)
# Calculate the overall average
overall_average = np.round(np.mean(total_values[mask]),2)
overall_average_median = np.round(np.mean(total_values_median[mask]),2)


#%%
for key, values in data.items():
    data[key] = [float('{:.3}'.format(value)) for value in values]
    
max_length = max(len(lst) for lst in data.values())


max_value = max(series.max() for series in nb_countries.values())

aligned_data_dict = {}
for key, lst in data.items():
    av = round(np.mean(lst),1)
    median = round(np.median(lst),1)
    aligned_data_dict[key] = np.append( np.append(np.append(np.round(lst,2),([np.nan] * (max_length - len(lst)))),av),median)
# Create a DataFrame from the aligned dictionary
aligned_data_dict2 = {}
for key, lst in nb_countries.items():
    av = lst.max() 
    median = lst.max() 
    aligned_data_dict2[key] = np.append( np.append(np.append(np.round(lst,2),([np.nan] * (max_length - len(lst)))),av),median)

df1 = pd.DataFrame(aligned_data_dict)
df1['average_across_variants'] = np.nan
df1['average_across_variants'].iloc[len(df1) - 2:] = [overall_average, overall_average_median]
df2 = [x for x in range(14, 197, 14)]
df3 = pd.DataFrame(aligned_data_dict2)
df3['average_across_variants'] = np.nan
df3['average_across_variants'].iloc[len(df1) - 2:] = [max_value, max_value]


if chosen_metric == "MAE2":
    df2.append("MMedAE")
    df2.append("MedMedAE")
else:
    df2.append("MMAE")
    df2.append("MedMAE")

# Visualize the matrix
x_tick_colors = [plt.cm.viridis(0.5) if col is None else plt.cm.viridis(0.75) for col in df1.iloc[0].values]

plt.figure(figsize=(20, 16))
x = 'Reds'

plt.imshow(df3.transpose(), cmap=x, aspect='auto', interpolation='none')
dat = np.array(df1.transpose())
for i in range(len(dat)):
    for j in range(len(dat[i])):
        if not np.isnan(dat[i, j]) and i!=32:
            plt.text(j, i, dat[i, j], ha='center', va='center', color='black', fontsize=16)
        else:
            plt.text(j, i, dat[i, j], ha='center', va='center', color='white', fontsize=16)
            



cbar = plt.colorbar(orientation='vertical', location = 'right')
cbar.set_label("Number of countries to predict for", fontsize=16)
cbar.ax.tick_params(labelsize=16)
plt.xlabel('Days of existence before total prevalence', fontsize=18)
plt.ylabel('Variant', fontsize=18)
plt.title(f'{directory_name}  {chosen_metric[:-1]} Performance Over Time', fontsize=20) # 
plt.xticks(range(len(df2)),df2,rotation=45, fontsize=16)
plt.yticks(range(len(df1.columns)), df1.columns, fontsize=16)
plt.show()

