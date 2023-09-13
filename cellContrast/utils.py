
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.neighbors import KDTree
from collections import Counter
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from scipy.stats import spearmanr
import torch



# set the device
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

    

def getModelSimMat(rep_x, rep_y):
            
    
    similarity_matrix = cosine_similarity(rep_x, rep_y)
    sorted_similarity_matrix = []
    index_matrix = []

    for i in range(similarity_matrix.shape[0]):
        sorted_idx = np.argsort(similarity_matrix[i,:])[::-1]
        sorted_row = similarity_matrix[i,sorted_idx]
        sorted_similarity_matrix.append(sorted_row)
        index_matrix.append(sorted_idx)
        
    sorted_similarity_matrix = np.asarray(sorted_similarity_matrix)
    index_matrix = np.asarray(index_matrix)

    
    return sorted_similarity_matrix,index_matrix,similarity_matrix


def infer_representations(model,query_adata):
    
    
    print("device",device)
    if(sparse.issparse(query_adata.X)):
        input_gene_exp = np.asarray(query_adata.X.todense())
    else:
        input_gene_exp = query_adata.X
        
    torch.tensor(input_gene_exp).float().to(device)
    model.eval()
    representation, projection = model(torch.tensor(input_gene_exp).float().to(device))
    if(dev!="cpu"):
        representation = representation.cpu()
        
    return representation.detach().numpy()


def calNeiborHit(truth_sorted_ind, predicted_sorted_ind):
    neighbor_hit_details = {'test_id':list(range(predicted_sorted_ind.shape[0]))}
    
    for k in range(20,220,20):
        
        cur_name = "neighbor_hit_k_"+str(k)
        neighbor_hit_details[cur_name] = [0]*predicted_sorted_ind.shape[0]
        
        for i, cur_truth_all_ind in enumerate(truth_sorted_ind):
            # ignor self
            cur_truth_neighbor_ind = set(cur_truth_all_ind[1:k+1])
            cur_predicted_neighbors = set(predicted_sorted_ind[i][1:k+1])
            
            overlapped_neighbors = cur_truth_neighbor_ind.intersection(cur_predicted_neighbors)
            overlapped_num = len(overlapped_neighbors)
            neighbor_hit_details[cur_name][i] = overlapped_num
            
    
    neighbor_hit_details = pd.DataFrame(neighbor_hit_details)
    return neighbor_hit_details


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


