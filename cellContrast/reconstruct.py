from argparse import ArgumentParser, SUPPRESS
import scanpy as sc
import cellContrast.train
import cellContrast.inference
import  logging
import sys
import os

logging.getLogger().setLevel(logging.INFO)






def formatInputAnn(args):
    
    '''
    
    
    
    '''
    train_adata = sc.read_h5ad(args.train_data_path)
    query_adata = sc.read_h5ad(args.query_data_path)
    
    # get the overlap genes of reference and query AnnData
    train_genes = train_adata.var_names
    query_genes = query_adata.var_names
    overlapped_genes = list(set(train_genes).intersection(set(query_genes)))
    if(len(overlapped_genes)<=0):
        sys.exit("[ERROR] 0 overlapped gene found between the training and query data.")
    
    logging.info("%s overlapped genes found between the training and query data" % (str(len(overlapped_genes))))
    
    train_gene_indices = [train_adata.var_names.get_loc(gene) for gene in overlapped_genes]
    formatted_train_adata = train_adata[:, train_gene_indices]
    
    query_gene_indices = [query_adata.var_names.get_loc(gene) for gene in overlapped_genes]
    formatted_query_adata = query_adata[:, query_gene_indices]
    
    return formatted_train_adata,formatted_query_adata

def main():
    
    # `reconstrcut.py` combines `train.py` and `inference.py` into one step
    
    parser = ArgumentParser(description="Train a cellContrast model")
    
    # training arguments
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
    
    
    # inference arugments
    parser.add_argument('--query_data_path', type=str,
                        help="The path of querying data with h5ad format (annData object)")
    parser.add_argument('--enable_denovo', action="store_true",help="(Optional) generate the coordinates de novo by MDS algorithm",default=False)
    parser.add_argument('--save_path',type=str,help="Save path of the spatial reconstructed SC data",default="./reconstructed_sc.h5ad")
    
    
    
    args = parser.parse_args()
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)
    
    # check arguments
    if(not os.path.exists(args.train_data_path)):
        logging.error("train data not exists!")
        sys.exit(1)
    if(not os.path.exists(args.query_data_path)):
        logging.error("query data not exists!")
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
    
    # format input training and query data
    train_adata, query_adata = formatInputAnn(args)
    
    # training the cellContrast model
    model = cellContrast.train.train_model(args,train_adata)
    
    # reconstruct the spatial relationships of query data
    logging.info("Performing spatial inference for the query data")
    reconstructed_query_adata = cellContrast.inference.perform_inference(query_adata,train_adata,model,args.enable_denovo)
    reconstructed_query_adata.write(args.save_path)
    
    
