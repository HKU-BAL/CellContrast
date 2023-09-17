import scanpy as sc
import anndata as ad
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
import random
import numpy as np
from random import choice
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse





def calLocationDistance(train_data,test_data):
    
    
    train_coor = np.column_stack((train_data.obs['x'].values,train_data.obs['y'].values))
    test_coor = np.column_stack((test_data.obs['x'].values,test_data.obs['y'].values))
    train_coor_tree = KDTree(train_coor,leaf_size=2)
    dist,ind  = train_coor_tree.query(test_coor,k=train_coor.shape[0])
    
    
    return dist,ind


def checkNeighbors(cur_adata,neighbor_k):
    
    '''
    return dist,ind of positive samples.    
    '''
    print("checkNeighbors.............")
    
    cur_coor = np.column_stack((cur_adata.obs['x'].values,cur_adata.obs['y'].values))
    cur_coor_tree = KDTree(cur_coor,leaf_size=2)
    location_dist,location_ind  = cur_coor_tree.query(cur_coor,k=(neighbor_k+1))
    location_dist = location_dist[:,1:]
    location_ind = location_ind[:,1:]
    
    return location_dist,location_ind




def loadTrainData(adata, neighbor_k, sample_ID_name='embryo'):
    
    '''
    '''
    
    if(sample_ID_name in adata.obs):
        unique_samples = adata.obs[sample_ID_name].unique()
    
    
    train_sample_number = unique_samples.shape[0]
    
    train_rep = []
    train_coors_mat = []
    # a list of dictionary
    pos_info = []
    
    
    # generate training dataset
    for cur_sample in unique_samples:
        
    
        # Filter the cells corresponding to the current embryo
        cur_sample_adata = adata[adata.obs[sample_ID_name] == cur_sample]
        
        # check if the input is sparse matrix
        if(issparse(cur_sample_adata.X)):
            train_input =  np.asarray(cur_sample_adata.X.todense())
        else:
            train_input = cur_sample_adata.X

        cur_train_rep = np.nan_to_num(train_input)
        
        # generate positive pair information
        pos_dist, pos_ind = checkNeighbors(cur_sample_adata,neighbor_k)
        cur_pos_info = {'pos dist':pos_dist,'pos ind':pos_ind}
        
        train_rep.append(cur_train_rep)
        pos_info.append(cur_pos_info)
        cur_train_coors_mat = np.column_stack((cur_sample_adata.obs['x'],cur_sample_adata.obs['y']))
        train_coors_mat.append(cur_train_coors_mat)
    
    
    return train_rep, train_coors_mat, pos_info




def loadBatchData(train_data_mat,train_coors_mat,batch_size,pos_info):
    '''
    
    generate batch training data
    
    '''
    
    train_pos_dist = pos_info['pos dist']
    train_pos_ind = pos_info['pos ind']
    
    train_index_list = list(range(train_data_mat.shape[0]))
    random.shuffle(train_index_list)
    train_data_size = train_data_mat.shape[0]

    half_batch_size =  int(batch_size/2)
    batch_num = train_data_size//half_batch_size
    
    for i in range(batch_num):
        
        start = i*half_batch_size
        end = start + half_batch_size
        
        tmp_index_list =  list(train_index_list[start:end])
       
        pos_peer_index = []

        neighbor_index = np.zeros((batch_size,batch_size))
        
        count = 0
        pos_index_list = []
        for j in tmp_index_list:
             
             cur_pos_peer_index = np.copy(train_pos_ind[j])
             random.shuffle(cur_pos_peer_index)
             
             pos_index_list.append(cur_pos_peer_index[0])
             
             neighbor_index[count][half_batch_size+count] = 1 
             neighbor_index[half_batch_size+count][count] = 1
 
             count += 1
     

        tmp_index_list.extend(pos_index_list)
        cur_index_list = np.asarray(tmp_index_list)
        cur_batch_mat = np.take(train_data_mat,cur_index_list,axis=0)
        cur_coor_mat = np.take(train_coors_mat,cur_index_list,axis=0)
        
        
        yield cur_batch_mat,neighbor_index,cur_coor_mat,cur_index_list
    pass
















