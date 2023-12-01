import scanpy as sc
from cellContrast.model import *
from cellContrast import inference

from cellContrast import utils
from argparse import ArgumentParser, SUPPRESS
import os
import json
import sys
from scipy import sparse
from sklearn.manifold import MDS
import numpy as np
import  logging
from scipy.spatial.distance import cdist
logging.getLogger().setLevel(logging.INFO)



# set the device
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def evaluate(query_adata):

    logging.info("Evaluation...")
    
    enable_jsd = True
    if("cell_type" not in query_adata.obs):
        logging.info("Cell type info not found, disable JSD evaluation")
        enable_jsd = False
    

    cur_truth_coor = np.column_stack((query_adata.obs['x'].values,query_adata.obs['y'].values))
    truth_distances =  cdist(cur_truth_coor, cur_truth_coor)
    truth_sorted_ind = np.argsort(truth_distances)

    precited_dist_mat = 1 - query_adata.uns['cosine sim of rep']
    predicted_sorted_ind = np.argsort(precited_dist_mat)
    predicted_referenced_coors = np.column_stack((query_adata.uns['referenced x'],query_adata.uns['referenced y']))
    predicted_coors_distance = cdist(predicted_referenced_coors, predicted_referenced_coors)
    
    
    # the nearest neighbor hit
    logging.info("evaluate avergae neighbor hit number")
    cur_neighbor_res = utils.calNeiborHit(truth_sorted_ind,predicted_sorted_ind)
    res_summary_all = cur_neighbor_res
   
    # Jessen-Shannon Distance
    if(enable_jsd):
        logging.info("evaluate average Jessen-Shannon Distance")
        cur_truth_cell_types = query_adata.obs['cell_type'].values
        cur_jsd_list = utils.calJSD(truth_sorted_ind,predicted_sorted_ind,cur_truth_cell_types,neighbor_num=20)
        res_summary_all['Cell type'] = cur_truth_cell_types
        res_summary_all["jsd"] = cur_jsd_list
        
    # Spearman's rank correlation coefficient
    logging.info("evaluate average spearman's rank correlation coefficient")
    spearman_corr_list,spearman_p_list = utils.calSpearmanRank(predicted_coors_distance,truth_distances)
    
    res_summary_all["spearman corr"] = spearman_corr_list
    
    logging.info("Evaluation DONE!")
    return res_summary_all

def main():
    
    parser = ArgumentParser(description="Evaluation of spatial reconstruction")
    
    parser.add_argument('--query_data_path', type=str,
                        help="The path of querying data with h5ad format (annData object)")
    parser.add_argument('--ref_data_path', type=str,
                        help="The path of querying data with h5ad format (annData object)")
    
    parser.add_argument('--model_folder', type=str,default="./cellContrast_models",
                        help="Save folder of model related files, default:'./cellContrast_models'")
    parser.add_argument('--parameter_file_path', type=str,
                        help="Please take the parameter file you used in the training phase,\
                        default:'./parameters/parameters_spot.json'",default="./parameters/parameters_spot.json")
    
    parser.add_argument('--save_path',type=str,help="Save path of evaluation result",default="./result.csv")
    args = parser.parse_args()
    
    
    # load params
    with open(args.parameter_file_path,"r") as json_file:
        params = json.load(json_file)
    
    
    # load models
    model, train_genes = inference.load_model(args,params)
    model.to(device)
    print("model",model)
    
    
    # inference
    ## check if the train genes exists in query data
    query_adata = sc.read_h5ad(args.query_data_path)
    query_adata = inference.format_query(query_adata,train_genes)
    
    ref_adata = sc.read_h5ad(args.ref_data_path)
    ref_adata = inference.format_query(ref_adata,train_genes) 
    
    
    reconstructed_query_adata = inference.perform_inference(query_adata,ref_adata,model,False)
    print("reconstructed query data",reconstructed_query_adata)
    
    res_all = evaluate(reconstructed_query_adata)
    for colName, colData in res_all.items():
        if(colName in ["test_id","Cell type"]):
            continue
        print("Average "+colName,np.mean(colData))
    res_all.to_csv(args.save_path)
  
    
    

if __name__ == '__main__':
    
    pass