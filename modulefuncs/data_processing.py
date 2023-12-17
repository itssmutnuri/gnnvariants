import numpy as np
import pandas as pd

from torch_geometric_temporal.signal import StaticGraphTemporalSignal


def process_data(df, T, countries, all_variants, restrictions, edge_index, is_graph, use_r, use_S, dom_thresh, partial_edgeW_calc):

    dates = df['date'].unique()
    feat_mats = []
    target_mats = []
    edge_weights = []
    
    # Loop over each date and pangoLineage (each gives us a snapshot)
    for d in dates:
        pangos = df[(df['date'] == d)]
        pangos = pangos['pangoLineage'].unique()
      
        controls = restrictions[restrictions['date'] == d]
        
        if is_graph:
          EW = partial_edgeW_calc(df = controls)
        else:
          EW = np.ones(len(countries),len(countries))
        
        for pangoLineage in pangos: 
            p_index = all_variants.index(pangoLineage)
            
            # Create the feat_matrix and target_matrix
            if use_S:
              feat_matrix = np.zeros((len(countries), T+1))
            else:
              feat_matrix = np.zeros((len(countries), T))
              
            target_matrix = np.zeros((len(countries), 2))
            target_matrix[:,0] = -1 #Will never reach prevalence
            
            countries_dom = df[(df['pangoLineage'] == pangoLineage) & (df['prev'] > dom_thresh)]
            
            k1 = countries_dom.drop_duplicates(subset='country', keep = 'first')
            idx2 = (k1.date - d).dt.days
            idx2[idx2 < 0] =0
            #Check if date of domination has passed, if it did then 0, else calculate it.
            idx = [countries.index(ci) for ci in countries_dom['country'].unique()]
            target_matrix[idx,0] = idx2
            target_matrix[idx,1] = 1 
            
            
            prev_values = df[(df['date'] >= (pd.Timestamp(d) - pd.DateOffset(weeks = (T*2)-1))) &
                             (df['date'] <= d) &
                             (df['pangoLineage'] == pangoLineage)]
            countries_pres = prev_values['country'].unique()
            
            if len(countries_pres)==0:
                continue
                
            for c in countries_pres:
                temp_c = prev_values[(prev_values['country'] == c)].sort_values(by='date')
                
                two_week_intervals = pd.date_range(end=pd.Timestamp(d), periods=T, freq='2W-MON')
                
                temp_c = temp_c[temp_c['date'].isin(two_week_intervals)]
                
                if use_r:
                  prev_values_c = temp_c['prev'].values
                else:
                  prev_values_c = temp_c['prevsO'].values
                
                
                # If no prev values were found, fill the row with 0s (It isn't in given country yet)
                if len(prev_values_c) == 0:
                    prev_values_c = np.zeros(T)
                    si = 0
                # If not enough prev values were found, pad with 0s
                elif len(prev_values_c) < T:
                    prev_values_c = np.pad(prev_values_c, (T-len(prev_values_c), 0), 'constant')
                    si = temp_c['S'].values[-1]
                else:
                    si = temp_c['S'].values[-1]
              
                if use_r:
                  log_prev_vals = np.log(prev_values_c + (10**-10))
                else:
                  log_prev_vals = prev_values_c
                  
                if use_S:
                  appended_prev_si = np.append(log_prev_vals, si)
                else:
                  appended_prev_si = log_prev_vals
                
                row_index = countries.index(c)

                feat_matrix[row_index, : ] = appended_prev_si
                
                
                # Get the days_to_prev value for the current date and pangoLineage
                target_vals = prev_values[(prev_values['date'] == d) & (prev_values['country'] == c)]
                
                days_to_prev = target_vals['days_to_prev'].values
                dom =  target_vals['B'].values
                # If no days_to_prev value was found, skip (doesn't become prevalent)
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

def process_data_test(df, T, d, countries, all_variants, restrictions, edge_index, is_graph, use_r, use_S, dom_thresh, partial_edgeW_calc):

    feat_mats = []
    target_mats = []
    edge_weights = []

    pangos = df['pangoLineage'].unique()
    controls = restrictions[restrictions['date'] == d] #We are just concerned with evaluating at this timestep.
    
    EW = partial_edgeW_calc(df = controls)
    
    for pangoLineage in pangos: #pangoLineages:
        p_index = all_variants.index(pangoLineage)

        # Create the feat_matrix and target_matrix
        if use_S:
          feat_matrix = np.zeros((len(countries), T+1))
        else:
          feat_matrix = np.zeros((len(countries), T))
          
        target_matrix = np.zeros((len(countries), 2))
        target_matrix[:,0] = -1 #Will never reach prevalence
        countries_dom = df[(df['pangoLineage'] == pangoLineage) & (df['prev'] > dom_thresh)]
        
        k1 = countries_dom.drop_duplicates(subset='country', keep = 'first')
        idx2 = (k1.date - d).dt.days
        idx2[idx2 < 0] =0
        #Check if date of dom has passed, if it did then 0, else calculate it.
        idx = [countries.index(ci) for ci in countries_dom['country'].unique()]
        target_matrix[idx,0] = idx2
        target_matrix[idx,1] = 1 

        prev_values = df[(df['date'] >= (pd.Timestamp(d) - pd.DateOffset(weeks = (T*2)-1))) &
                         (df['date'] <= d) &
                         (df['pangoLineage'] == pangoLineage)]
        countries_pres = prev_values['country'].unique()
        
        if len(countries_pres)==0:
            continue
        for c in countries_pres:
            temp_c = prev_values[(prev_values['country'] == c)].sort_values(by='date')
            two_week_intervals = pd.date_range(end=pd.Timestamp(d), periods=T, freq='2W-MON')
            temp_c = temp_c[temp_c['date'].isin(two_week_intervals)]
            
            if use_r:
              prev_values_c = temp_c['prev'].values
            else:
              prev_values_c = temp_c['prevsO'].values

            # If no prev values were found, fill the row with 0s
            if len(prev_values_c) == 0:
              prev_values_c = np.zeros(T)
              si = 0
            elif len(prev_values_c) < T:
              prev_values_c = np.pad(prev_values_c, (T-len(prev_values_c), 0), 'constant')
              si = temp_c['S'].values[-1]
            else:
              si = temp_c['S'].values[-1]
                  
            if use_r:
              log_prev_vals = np.log(prev_values_c + (10**-10))
            else:
              log_prev_vals = prev_values_c
              
            if use_S:
              appended_prev_si = np.append(log_prev_vals, si)
            else:
              appended_prev_si = log_prev_vals
              
            row_index = countries.index(c)

            feat_matrix[row_index, : ] = appended_prev_si
            
            
            # Get the days_to_prev value for the current date and pangoLineage
            target_vals = prev_values[(prev_values['date'] == d) & (prev_values['country'] == c)]
            
            days_to_prev = target_vals['days_to_prev'].values
            dom =  target_vals['B'].values
            # If no days_to_prev value was found, skip it (doesn't become prevalent)
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