import numpy as np
import torch

def edgeW_calc(df, countries, adj_mat, EW_bool, edge_index):
    weighted_mat = np.ones((len(countries),len(countries)))
      

    # sort by restriction lvls 0->4
    df = df.sort_values(by='international_travel_controls')
    
    #Calculate weights (inverse scale restriction lvls [0.1 to 1])
    for c in df['country'].unique():
        original_value = df[df['country'] ==c]['international_travel_controls'].values[0]
        scaled_value = 1 - (original_value / 4) * 0.9
        indx = np.where(countries == c)
        weighted_mat[indx, :] = scaled_value
        weighted_mat[:,indx] = scaled_value
    
    
    #Set all to one (unweighted)
    if not EW_bool: #IF using edge weights, skip
      weighted_mat = np.ones(weighted_mat.shape)
      
      
    #Mask the unnecessary ones  
    weighted_mat = np.where(adj_mat == 0, 0, weighted_mat) 
    for i in range(len(countries)):
      weighted_mat[i, i] = 1
    
    weight_tensor = torch.DoubleTensor(weighted_mat)
    
    edge_weights = []
    for e in range(len(edge_index)):
        edge_idx = edge_index[e]
        i = edge_idx[0]
        j = edge_idx[1]
        edge_weights.append(weight_tensor[i][j])
        
    edge_weights = torch.DoubleTensor(edge_weights)
    
    return edge_weights