from torch.utils.data import Dataset
import tqdm
import torch
import random
from omegaconf import OmegaConf
from model.msa_read import get_esm_msa_rep
from model.fasta_read import get_seqlen_from_fasta
from model.pdb_read import get_coords_seq
from model.mutant_read import get_splited_data
from model.mutant_read import get_mutant_data
from model.mutant_read import get_mut_seq
from model.ProteinGraph import _get_wt_graph
from model.ProteinGraph import _get_mutant_graph

class GVPdataset(Dataset):
    
    """path_prefix: input file (two L: sequence; type) dataset_names """

    def __init__(self, path_prefix, dataset_name, dataset_config,mut_prefix,folder_num=0, mode='train', device='cuda:0'):
        super(GVPdataset, self).__init__()

        self.msa_file = '{}/{}/{}.a2m'.format(path_prefix,dataset_name,dataset_name)
        self.fasta_file = '{}/{}/{}.fasta'.format(path_prefix,dataset_name,dataset_name)
        self.pdb_file = '{}/{}/{}.pdb'.format(path_prefix,dataset_name,dataset_name)
        self.mutant_file = '{}/{}/{}{}.csv'.format(path_prefix,dataset_name,dataset_name,mut_prefix)
        self.device = device
        self.dataset_name = dataset_name
        self.data_config = OmegaConf.load(dataset_config)
        self.msa_rep = get_esm_msa_rep(self.msa_file,num_seqs=256,device='cuda:0') # model/msa_read.py
        #print('==== msa readin finished')
        self.seqlen, self.wt_seq, self.offset = get_seqlen_from_fasta(self.fasta_file) # model/fasta_read.py
        #print('==== fasta readin finished')
        self.coords_binds_pad, self.seq_bind_pad = get_coords_seq(self.pdb_file,self.data_config[dataset_name],ifbindchain=True,ifbetac=False) # model/pdb_read.py
        #print('==== pdb readin finished')

        if mode == 'train':
            self.data_df = get_splited_data(path_prefix,dataset_name,data_split_method=0)[folder_num][0] # model/mutant_read.py #folder_num fold random mode=train 0
        elif mode == 'val':
            self.data_df = get_splited_data(path_prefix,dataset_name,data_split_method=0)[folder_num][1] # model/mutant_read.py #folder_num fold random mode=train 0
        elif mode == 'test':
            self.data_df = get_mutant_data(path_prefix,dataset_name,data_split_method=0)
 
        #print('==== mutant readin finished')
        self.data_df = get_mut_seq(self.data_df,self.wt_seq,'msa_seq',self.offset) # model/mutant_read.py
        self.data_df = get_mut_seq(self.data_df,self.seq_bind_pad,'coords_seq',self.offset) # model/mutant_read.py
        #print('==== mutant template_seq and diff_chain_seq readin finished')

        CHARS = ["-", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L","M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

        self.letter_to_num = {c: i for i, c in enumerate(CHARS)}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

        self.additional_node = len(self.seq_bind_pad)-len(self.wt_seq)
        self.wt_graph = _get_wt_graph(self.data_df,self.coords_binds_pad,self.msa_rep,self.additional_node,
                                      self.letter_to_num,self.device,get_msa_info=True,if_category=False) # model/proteingraph_read.py

    # file lines             
    def __len__(self):
        return len(self.data_df)
    
    # return dict to tensor
    def __getitem__(self,i):
        line = self.data_df.iloc[i]
        graph = _get_mutant_graph(line,self.offset,self.letter_to_num,self.device,if_category=False) # model/proteingraph_read.py
        return graph, self.wt_graph

##############################################################################################

class GVP_dataset_all(Dataset):

    """path_prefix: input file (two L: sequence; type) dataset_names """

    def __init__(self,data_df,coords_binds_pad,msa_rep,additional_node,offset,device='cuda:0'):
        super(GVP_dataset_all, self).__init__()

        self.data_df = data_df
        self.coords_binds_pad = coords_binds_pad
        self.msa_rep = msa_rep
        self.additional_node = additional_node
        self.offset = offset
        self.device = device

        CHARS = ["-", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L","M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

        self.letter_to_num = {c: i for i, c in enumerate(CHARS)}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

        self.wt_graph = _get_wt_graph(self.data_df,self.coords_binds_pad,self.msa_rep,self.additional_node,
                                      self.letter_to_num,self.device,get_msa_info=True,if_category=False) # model/proteingraph_read.py

    def __len__(self):
        return len(self.data_df)
    
    # return dict to tensor
    def __getitem__(self,i):
        line = self.data_df.iloc[i]
        graph = _get_mutant_graph(line,self.offset,self.letter_to_num,self.device,if_category=False) # model/proteingraph_read.py
        return graph, self.wt_graph


class Dataset_in(object):
    
    """path_prefix: input file (two L: sequence; type) dataset_names """

    def __init__(self,path_prefix,dataset_name,dataset_config,mut_prefix,folder_num=0,mode='train',device='cuda:0'):

        self.msa_file = '{}/{}/{}.a2m'.format(path_prefix,dataset_name,dataset_name)
        self.fasta_file = '{}/{}/{}.fasta'.format(path_prefix,dataset_name,dataset_name)
        self.pdb_file = '{}/{}/{}.pdb'.format(path_prefix,dataset_name,dataset_name)
        self.mutant_file = '{}/{}/{}{}.csv'.format(path_prefix,dataset_name,dataset_name,mut_prefix)
        self.device = device
        self.path_prefix = path_prefix
        self.dataset_name = dataset_name
        self.folder_num = folder_num
        self.mut_prefix = mut_prefix

        self.data_config = OmegaConf.load(dataset_config)
        self.msa_rep = get_esm_msa_rep(self.msa_file,num_seqs=256,device='cuda:0') # model/msa_read.py
        #print('==== msa readin finished')
        self.seqlen, self.wt_seq, self.offset = get_seqlen_from_fasta(self.fasta_file) # model/fasta_read.py
        #print('==== fasta readin finished')
        self.coords_binds_pad, self.seq_bind_pad = get_coords_seq(self.pdb_file,self.data_config[dataset_name],ifbindchain=True,ifbetac=False) # model/pdb_read.py
        #print('==== pdb readin finished')
        self.additional_node = len(self.seq_bind_pad)-len(self.wt_seq)
        self.mode = mode

    def get_graphset(self):

        if self.mode == 'train':
            self.data_df,self.val_df = get_splited_data(self.path_prefix,self.dataset_name,data_split_method=0,suffix = self.mut_prefix)[self.folder_num] #mutant_read.pysplit random
            self.data_df = get_mut_seq(self.data_df,self.wt_seq,'msa_seq',self.offset) # model/mutant_read.py
            self.data_df = get_mut_seq(self.data_df,self.seq_bind_pad,'coords_seq',self.offset) # model/mutant_read.py
            self.val_df = get_mut_seq(self.val_df,self.wt_seq,'msa_seq',self.offset) # model/mutant_read.py
            self.val_df = get_mut_seq(self.val_df,self.seq_bind_pad,'coords_seq',self.offset) # model/mutant_read.py
            train = GVP_dataset_all(self.data_df,self.coords_binds_pad,self.msa_rep,self.additional_node,self.offset,self.device)
            val = GVP_dataset_all(self.val_df,self.coords_binds_pad,self.msa_rep,self.additional_node,self.offset,self.device)
            return train,val
            
        elif self.mode == 'test':
            self.test_df = get_mutant_data(self.path_prefix,self.dataset_name,data_split_method=0,suffix = self.mut_prefix)
            self.test_df = get_mut_seq(self.test_df,self.wt_seq,'msa_seq',self.offset) # model/mutant_read.py
            self.test_df = get_mut_seq(self.test_df,self.seq_bind_pad,'coords_seq',self.offset) # model/mutant_read.py
            test = GVP_dataset_all(self.test_df,self.coords_binds_pad,self.msa_rep,self.additional_node,self.offset,self.device)
            return test

