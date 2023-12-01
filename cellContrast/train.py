from argparse import ArgumentParser, SUPPRESS
import json
import sys
import os
import scanpy as sc
from cellContrast.model import *
import cellContrast.loadData as loadData
import torch
import numpy as np
from tqdm import tqdm
import time
import logging

logging.getLogger().setLevel(logging.INFO)

sample_field_name = 'embryo'

def adjust_learning_rate(optimizer, epoch,initial_lr,num_epochs):
    
    """Adjusts the learning rate based on the cosine annealing strategy."""
    
    lr = 0.5 * initial_lr * (1 + np.cos(np.pi * epoch / num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_model(args,model,params,optimizer,LOSS,train_genes):
    
    cur_save_path = os.path.join(args.save_folder,"epoch_"+str(params["training_epoch"])+".pt")
    
    torch.save({
                  'epoch': params['training_epoch'],
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': LOSS,
                  'params': params,
                  'train_genes':train_genes,
                   }, cur_save_path)
    
    


def train_model(args,train_adata=None):
    
    logging.info("Training cellContrast model")
    
    
    # load parameter settings
    with open(args.parameter_file_path,"r") as json_file:
        params = json.load(json_file)
    print("parameters",params)
    
    
    
    # load data if necessary
    if(not train_adata):
        logging.info("Load training data")
        train_adata = sc.read_h5ad(args.train_data_path)
    
    if(sample_field_name in train_adata.obs):
        train_sample_number = train_adata.obs[sample_field_name].unique().shape[0]
    else:
        train_adata.obs[sample_field_name] = 'sample_1'
        train_sample_number  = train_adata.obs[sample_field_name].unique().shape[0]
    

    train_genes = train_adata.var_names
    
    # init the training data information
    train_data_mat, train_coors_mat, pos_info = loadData.loadTrainData(adata=train_adata,neighbor_k=params['k_nearest_positives'])    
    
    n_input = train_data_mat[0].shape[1]
    params['n_input'] = n_input
    
    
    # set the device
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    logging.info("Using device %s" %(dev))
    
    device = torch.device(dev)
    
    
    # init the model
    model = CellContrastModel(n_input=params['n_input'],\
                              n_encoder_hidden=params["n_encoder_hidden"],n_encoder_latent=params["n_encoder_latent"],\
                              n_encoder_layers=params["n_encoder_layers"],n_projection_hidden=params["n_projection_hidden"],\
                              n_projection_output=params["n_projection_output"])
    model.to(device)
    l = ContrastiveLoss(temperature=params['temperature'])
    
    print(model)
    
    # Set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=params['inital_learning_rate'],
                                momentum=0.9, weight_decay=5e-4)
    pbar = tqdm(total=params['training_epoch'], desc="Loss: 0.0000")

    for cur_epoch in range(0,params['training_epoch']):
        
       
        cur_lr = adjust_learning_rate(optimizer, cur_epoch, params['inital_learning_rate'], params['training_epoch'])
        
        total_loss,total_num = 0.0,0.0
        
        for cur_sample_idx in range(train_sample_number):
            
            # When there are multiple sample, train samples separately
            
            cur_train_data_mat = train_data_mat[cur_sample_idx]
            cur_train_coors_mat = train_coors_mat[cur_sample_idx]
            cur_pos_info = pos_info[cur_sample_idx]
            
            for gene_profile,positive_index,coor_mat,_ in loadData.loadBatchData(cur_train_data_mat,cur_train_coors_mat,params['batch_size'],cur_pos_info):
                
                input_gene_exp = torch.tensor(gene_profile).float().to(device) 
                representation, projection = model(input_gene_exp)
                
                # compute the loss
                loss = l(projection,torch.tensor(positive_index).to(device))
                # optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()*params['batch_size']
                total_num += gene_profile.shape[0]
                pass
        LOSS = total_loss/total_num
        pbar.update(1)
        pbar.set_description(f"Loss: {LOSS:.3f}")
        time.sleep(0.1)
        pass
    pbar.close()
    
    print("LOSS",LOSS)
    
    # Save the last model
    save_model(args,model,params,optimizer,LOSS,train_genes)
    return model
    
    
    



def main():
    
    parser = ArgumentParser(description="Train a cellContrast model")
    
    parser.add_argument('--train_data_path', type=str,
                        help="The path of training data with h5ad format (annData object)")
    
    parser.add_argument('--save_folder', type=str,
                        help="Save folder of model related files, default:'./cellContrast_models'",default="./cellContrast_models")
    
    parser.add_argument('--parameter_file_path', type=str,
                        help="Path of parameter settings, customize it based on reference ST\
                        default:'./parameters/parameters_spot.json'",default="./parameters/parameters_spot.json")
    
    parser.add_argument('-sc','--single_cell',\
                        help="default:false, set this flag will swithing to the single-cell resolution ST mode, which uses the predefined './parameters/parameters_singleCell.json'",\
                        action='store_true')
    
    
    
    args = parser.parse_args()
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)
    
    # check arguments
    if(not os.path.exists(args.train_data_path)):
        print("train data not exists!")
        sys.exit(1)
    
    # check the parameter files
    if(args.single_cell):
        # change the parameter settings to the single-cell mode unless users have customized it.
        args.parameter_file_path = "./parameters/parameters_singleCell.json"
        
    
    if(not os.path.exists(args.parameter_file_path)):
        print("parameter file not exists!")
        sys.exit(1)
    
    if(not os.path.exists(args.save_folder)):
        os.mkdir(args.save_folder)
    
    
    train_model(args)
    pass


if __name__ == '__main__':
    
    main()