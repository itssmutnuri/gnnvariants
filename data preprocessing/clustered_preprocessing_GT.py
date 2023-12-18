import pandas as pd
import numpy as np
import json
import os
import math
import difflib
from csaps import csaps
from matplotlib import pyplot as plt
# Directory path containing the JSON files
directory = 'data/clustered_new'
#%% supporting functions
def checkCountries(c1,c2):
    for c in c1:
        if c in c2:
            continue
        else:
            print(c, difflib.get_close_matches(c, c2))
#%%
# List to store the data
data = []
# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        filepath = os.path.join(directory, filename)
        # Read the JSON file
        with open(filepath, 'r') as file:
            json_data = json.load(file)
            var = filename.split("_")[0]
            # Extract the attributes from the JSON
            for country, attributes in json_data.items():
                week = attributes['week']
                total_sequences = attributes['total_sequences']
                cluster_sequences = attributes['cluster_sequences']
                # Create a row for each week
                for i in range(len(week)):
                    data.append({
                        'Variant': var,
                        'Country': country,
                        'Week': week[i],
                        'Total Sequences': total_sequences[i],
                        'Cluster Sequences': cluster_sequences[i]
                    })
# Create a dataframe from the collected data
df = pd.DataFrame(data)

#%% dataset alignment            

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
    
variants = df #3rd

countries = list(set(variants['Country']))

#Make sure countries are spelt the same between variants and airport dataset
checkCountries(countries,IATA_country[:,0])

# Airport info N/A (16): Andorra, Bonaire, Canary Islands, Curacao, Crimea, Saint Barthélemy,
# Timor-Leste, Sint Maarten, Democratic Republic of the Congo, Saint Martin, Monaco
# Kosovo, Sint Eustatius, Union of the Comoros, Liechtenstein, Eswatini, Republic of the Congo

#Union of the comoros, Liechtenstein, kosovom timor-leste
 
#Macedonia->North Macedonia
# IATA_country[:,0] = np.char.replace(IATA_country[:,0], "Macedonia", "North Macedonia")
# #Cote d'Ivoire -> Côte d'Ivoire
# IATA_country[:,0] = np.char.replace(IATA_country[:,0], "Cote d'Ivoire", "Côte d'Ivoire")
# #Cape Verde - > Cabo Verde
# IATA_country[:,0] = np.char.replace(IATA_country[:,0], "Cape Verde", "Cabo Verde")
#United States->USA
IATA_country[:,0] = np.char.replace(IATA_country[:,0], "United States", "USA")
route[:,0] = np.char.replace(route[:,0], "United States", "USA")
route[:,1] = np.char.replace(route[:,1], "United States", "USA")

countries = list(set(variants['Country']))
print("\n \n \n Replaced....")
#Check again
checkCountries(countries,IATA_country[:,0])
countires_NA= ["Bonaire", "Curacao", "Kosovo", "Sint Maarten"]

variants = variants[~variants['Country'].isin(countires_NA)]

#%% Get new prevalance (Compare to just two most dominant variants)  
#& save which 2 variants are most dominant in each country at time of S calc
var_prev_N = variants# pd.read_csv("splined21i_clustered.csv") 
var_prev_N['Week']= pd.to_datetime(var_prev_N['Week'])
var_prev_S = pd.DataFrame(columns=['pangoLineage', 'country', 'date', 'count', 'prev', 'days_to_prev', 'pangoLineage2', 'prevsO'])
count = 0
countries_var = list(set(var_prev_N.Country))
for c in countries_var:
    count = count + 1
    print(c,count)
    var_totals = var_prev_N[(var_prev_N.Country==c)]
    var_totals = var_totals.reset_index(drop=True)
    dates_totals = list(set(var_totals.Week))
    for d in dates_totals:
        v_d = var_totals[var_totals.Week == d]
        v_d = v_d.reset_index(drop=True)
        varList = list(set(v_d.Variant))
        if sum(pd.to_numeric(v_d['Cluster Sequences'])) == 0:
               continue
        if len(varList)==1:
            #So that way it doesn't go to infinity, we set a cap            
            prev = 100000
            days_to_prev = 0
            var_prev_S = var_prev_S.append({'pangoLineage' : varList[0],
                                        'country' : c, 'date' : d,'count':sum(pd.to_numeric(v_d['Cluster Sequences'])), 'prev': prev, 
                                        'days_to_prev': days_to_prev, 'pangoLineage2': "N/A", 'prevsO':1},ignore_index = True)
            continue
        
        total = sum(pd.to_numeric(v_d['Cluster Sequences'])) #np.mean(v_d.total)#
        vds = v_d.sort_values('Cluster Sequences')
        v1 = vds['Variant'].iloc[-1]
        v2 = vds['Variant'].iloc[-2]
        m = list(pd.to_numeric(vds['Cluster Sequences']))[-1]/total #max(pd.to_numeric(v_d['count']))
        m2 = list(pd.to_numeric(vds['Cluster Sequences']))[-2]/total
        for var in varList:           
            v_c_v = v_d[v_d.Variant == var]
            #prevailance
            prev =  sum(pd.to_numeric(v_c_v['Cluster Sequences']))/total
            prevs=prev
            #if prev < 0.05: #Ignore anything less than 5% prev (assume unassigned)
                #continue
            if prev == 1:
                prev = 100000
            else:
                if sum(pd.to_numeric(v_c_v['Cluster Sequences']))/total == m:
                    prev = sum(pd.to_numeric(v_c_v['Cluster Sequences']))/(total*m2)
                    var_prev_S = var_prev_S.append({'pangoLineage' : var,
                                                'country' : c, 'date' : d,'count': sum(pd.to_numeric(v_c_v['Cluster Sequences'])), 'prev': prev, 
                                                'days_to_prev': None, 'pangoLineage2': v2, 'prevsO':prevs},ignore_index = True)
                else:
                    prev =  sum(pd.to_numeric(v_c_v['Cluster Sequences']))/(total*m) #total 
                    var_prev_S = var_prev_S.append({'pangoLineage' : var,
                                                'country' : c, 'date' : d,'count': sum(pd.to_numeric(v_c_v['Cluster Sequences'])), 'prev': prev, 
                                                'days_to_prev': None, 'pangoLineage2': v1, 'prevsO':prevs},ignore_index = True)
                # prev = prev/(1-prev)
            
            

var_prev_S.to_csv("var_data_GTOG.csv",index=False)

#%% filter out pangos that dont reach 20% ever or atleast 3 other countries
varList = list(set(var_prev_S.pangoLineage))
# countries_var =  list(set(var_prev_S.country))
for var in varList:
    var_prev_G = var_prev_S[(var_prev_S.pangoLineage==var)]
    countries_var = list(set(var_prev_G.country))
    if len(countries_var) < 3:
        continue
    for c in countries_var:
        var_prev_SV = var_prev_G[(var_prev_G.country==c)]
        prevList = list(set(var_prev_SV.prev))
        if max(prevList) < 0.2:
            var_prev_S = var_prev_S[(var_prev_S.pangoLineage!=var) | (var_prev_S.country != c)]
        
var_prev_S.to_csv("splined_prevT21i_clusteredF_GTA.csv",index=False)   

#%%
import csv
with open(r"all_vars21_clustered.csv", mode = 'r') as file:
    reader = csv.reader(file)
    all_variants = []
    for row in reader:
        all_variants.append(row[0])