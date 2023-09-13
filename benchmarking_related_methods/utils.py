import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from scipy.stats import spearmanr

def calNeiborHit(truth_sorted_ind, predicted_sorted_ind):
    neighbor_hit_details = {'test_id':list(range(predicted_sorted_ind.shape[0]))}
    
    average_res = {'K nearest neighbor':[],'Average hit number':[]}
    
    for k in range(20,220,20):
        
        cur_name = "neighbor_hit_k_"+str(k)
        neighbor_hit_details[cur_name] = [0]*predicted_sorted_ind.shape[0]
        
        for i, cur_truth_all_ind in enumerate(truth_sorted_ind):
            
            cur_truth_neighbor_ind = set(cur_truth_all_ind[1:k+1])
            cur_predicted_neighbors = set(predicted_sorted_ind[i][1:k+1])
            
            overlapped_neighbors = cur_truth_neighbor_ind.intersection(cur_predicted_neighbors)
            overlapped_num = len(overlapped_neighbors)
            neighbor_hit_details[cur_name][i] = overlapped_num
            
        average_res['K nearest neighbor'].append(k)
        average_res['Average hit number'].append(np.mean(neighbor_hit_details[cur_name]))
    
    
    neighbor_hit_details = pd.DataFrame(neighbor_hit_details)
    return neighbor_hit_details,average_res

def get_jensenshannon(prediction_counts, truth_counts,unique_cell_names):
    
  
    prediction_datasets = []
    for cell_name in unique_cell_names:
        if(cell_name in prediction_counts):
            prediction_datasets.append(prediction_counts[cell_name])
        else:
            prediction_datasets.append(0)
            
    
    truth_datasets = []
    for cell_name in unique_cell_names:
        if(cell_name in truth_counts):
            truth_datasets.append(truth_counts[cell_name])
        else:
            truth_datasets.append(0)
    prediction_datasets = np.asarray(prediction_datasets)
    truth_datasets = np.asarray(truth_datasets)
    
    prediction_datasets = prediction_datasets/prediction_datasets.sum()
    truth_datasets = truth_datasets/truth_datasets.sum()
    
    jsd = jensenshannon(prediction_datasets, truth_datasets)
    
    return jsd

def calJSD(truth_sorted_ind,predicted_sorted_ind,all_cell_types,neighbor_num=20):
    
    '''
    
    predicted_sorted_ind: sorted index of predicted distances
    
    '''
    
    cell_type_set = set(all_cell_types)
    jsd_list = []
    
    for index, neighbor_idx in enumerate(predicted_sorted_ind):
        

        cur_truth_neighbor_idx = truth_sorted_ind[index][1:neighbor_num+1]
        cur_predict_neighbor_idx = neighbor_idx[1:neighbor_num+1]
        
        truth_neighbor_cell_types = np.take(all_cell_types,cur_truth_neighbor_idx)
        truth_counts = Counter(truth_neighbor_cell_types)
       
        predict_neighbor_cell_types = np.take(all_cell_types,cur_predict_neighbor_idx)
        predict_counts = Counter(predict_neighbor_cell_types)
        
        cur_jsd = get_jensenshannon(predict_counts, truth_counts,cell_type_set)
       
        jsd_list.append(cur_jsd) 
        
  
    return jsd_list



def calSpearmanRank(mat1, mat2):
    
    '''
    mat1: predicted pair-wise distances of cells
    mat2: truth pair-wise distances of cells
    
    '''
    
    spearman_corr = []
    spearman_p = []
    for i in tqdm(range(mat1.shape[0])):
        
        x = mat1[i]
        y = mat2[i]
    
        corr,pval = spearmanr(x,y)
        spearman_corr.append(corr)
        spearman_p.append(pval)
        
    return spearman_corr,spearman_p