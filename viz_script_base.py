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
#21H.Mu, #21G.Lambda, 21B.Kappa, 21A.Delta, 20J.Gamma.V3, 20H.Beta.V2, first 4 also
# Directory containing CSV files
csv_directory = r"C:\USC\Research\GNN_T4"
csv_directory2 = r"C:\USC\Research\baseline_new"
# User-defined metric
chosen_metric = 'MAE2'#'MAE2'


# Function to calculate biweekly dates
def biweekly_dates(date_list):
    temp = copy.deepcopy(date_list)
    for i in range(len(date_list)-1):
        gap = date_list[len(date_list) - i - 1] - date_list[len(date_list) - i - 2]
        if gap > timedelta(weeks=2):
            temp[0:len(date_list) - i - 1] = 0# [date_list[-1] + timedelta(days=100)]*(len(date_list) - i)
            break

    # Check if the largest gap is greater than 4 weeks
    # if largest_gap >= timedelta(weeks=4):
    #     # Return the second portion of the list
    #     date_list[0:largest_gap_index] = 0
        #return 

    # If there is no such gap, return an empty list
    return temp

def biweekly_dates_2(date_list):
    current_date = date_list.iloc[-1]
    print(date_list)
    dates = []
    dates.append(current_date)
    count = 0
    while current_date >= date_list.iloc[0] and count < 2:
        print("OK")
        current_date -= timedelta(weeks=2)
        #print(current_date)
        if not (current_date in date_list.values):
            #print(date_list)
            count+=1
            continue
        else:
            count = 0 #resets if not consecutive
        dates.append(current_date)
        print(dates)
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
        file_path2 = os.path.join(csv_directory2, csv_file[:-4] + "_X" + ".csv")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        if df['MAE2'].iloc[0] == -1:
            continue
        df_base = pd.read_csv(file_path2)
        # Calculate days till prevalence
        days_till_prevalence = calculate_days_till_prevalence(df['date'], df['MAE2'])
        
        # Store data     
        # tempp = df['date']
        index = biweekly_dates_2(df['date'])

        if csv_file == "21A.Delta.S.K417.csv":
            df_temp = df
            index_temp = index
        
        indices = [df.index[df['date'] == timestamp][0] for timestamp in index]
        temp = df.iloc[indices]
        temp = temp.iloc[:-1]
        temp_base = df_base.iloc[indices]
        temp_base = temp_base.iloc[:-1]
        
        data[variant] = temp_base[chosen_metric]
        temp2 = pd.Series((-1*np.array(days_till_prevalence))[indices])
        data_2[variant] = temp2[:-1]
        temp3 =  df['countries'].iloc[indices]
        temp3 = temp3.iloc[:-1]
        nb_countries[variant] = temp3.apply(eval).apply(len)
        
        if len(date_max) < len(data_2[variant]):
            date_max = data_2[variant]
        # Calculate and store the average performance over time
        average_performance = np.mean(temp_base[chosen_metric])
        averages[variant] = average_performance
        medians[variant] = np.median(temp_base[chosen_metric])
        total_values.append(average_performance)
        total_values_median.append(np.median(temp_base[chosen_metric]))

# Calculate the overall average
overall_average = np.round(np.nanmean(total_values),2)
overall_average_median = np.round(np.nanmean(total_values_median),2)

# Create a matrix with NaN values
#matrix = pd.DataFrame(data)
#%%
for key, values in data.items():
    data[key] = [float('{:.3}'.format(value)) for value in values]
    
max_length = max(len(lst) for lst in data.values())
#data1 = {key: value[::-1] for key, value in data.items()}

max_value = max(series.max() for series in nb_countries.values())
# for key, series in nb_countries.items():
#     nb_countries[key] = pd.concat([series, pd.Series([series.max(), series.max()])])
# Prepend shorter lists with NaN values
# aligned_data_dict = {key: (lst + [np.nan] * (max_length - len(lst))) }
aligned_data_dict = {}
for key, lst in data.items():
    av = round(np.nanmean(lst),1)
    median = round(np.nanmedian(lst),1)
    aligned_data_dict[key] = np.append( np.append(np.append(np.round(lst,2),([np.nan] * (max_length - len(lst)))),av),median)
# Create a DataFrame from the aligned dictionary
# df1 = pd.DataFrame(aligned_data_dict)
aligned_data_dict2 = {}
for key, lst in nb_countries.items():
    av = lst.max() #round(np.mean(lst),1)
    median = lst.max() #round(np.median(lst),1)
    aligned_data_dict2[key] = np.append( np.append(np.append(np.round(lst,2),([np.nan] * (max_length - len(lst)))),av),median)

df1 = pd.DataFrame(aligned_data_dict)
df1['average_across_variants'] = np.nan
df1['average_across_variants'].iloc[len(df1) - 2:] = [overall_average, overall_average_median]
df2 = [x for x in range(14, 197, 14)]
df3 = pd.DataFrame(aligned_data_dict2)
df3['average_across_variants'] = np.nan
df3['average_across_variants'].iloc[len(df1) - 2:] = [max_value, max_value]

#df2.reverse()
if chosen_metric == "MAE2":
    df2.append("MMedAE")
    df2.append("MedMedAE")
else:
    df2.append("MMAE")
    df2.append("MedMAE")
# Merge the DataFrames on the common keys
#merged_df = df1.join(df2, rsuffix='_df2')

# Reorder the columns
#merged_df = merged_df[sorted(merged_df.columns)]

# Fill NaN values with a placeholder (in this case, '-')
#merged_df = merged_df.fillna('-')

#%%
# Add a column for average performance
#matrix['Average'] = list(averages.values()) + [overall_average]

# Visualize the matrix
x_tick_colors = [plt.cm.viridis(0.5) if col is None else plt.cm.viridis(0.75) for col in df1.iloc[0].values]

plt.figure(figsize=(20, 16))
x = 'Reds'
# if chosen_metric == "f11":
#     x = 'viridis'
plt.imshow(df3.transpose(), cmap=x, aspect='auto', interpolation='none')
dat = np.array(df1.transpose())
for i in range(len(dat)):
    for j in range(len(dat[i])):
        if np.isnan(dat[i, j]) and np.isnan(df3.iloc[j,i]):
            plt.text(j, i, dat[i, j], ha='center', va='center', color='white', fontsize=16)
        elif i==32:
            plt.text(j, i, dat[i, j], ha='center', va='center', color='white', fontsize=16)
        else:
            plt.text(j, i, dat[i, j], ha='center', va='center', color='black', fontsize=16)

cbar = plt.colorbar(orientation='vertical', location = 'right')
cbar.set_label("Number of countries to predict for", fontsize=16)
cbar.ax.tick_params(labelsize=16)
plt.xlabel('Days of existence before total prevalence', fontsize=18)
plt.ylabel('Variant', fontsize=18)
plt.title(f'Baseline  C-Median Performance Over Time', fontsize=20) #{directory_name} {chosen_metric[:-1]}
plt.xticks(range(len(df2)),df2,rotation=45, fontsize=16)
plt.yticks(range(len(df1.columns)), df1.columns, fontsize=16)
plt.show()
