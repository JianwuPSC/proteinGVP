import os
import pandas as pd
import math
from sklearn.utils import shuffle
# from mutant.csv input 'mutant', 'log_fitness', 'seq', 'num_mut', 'dataset_name'
# split_method 0: split randomly , 1: site-specific,

def get_mutant_data(path_prefix,dataset_name,data_split_method=0,suffix = '',data_dir_prefix = ''):
    if data_split_method == 0:
        splitdatas = []
        datafile = '{}/{}/{}{}.csv'.format(path_prefix,dataset_name,dataset_name,suffix)
        alldata = pd.read_csv(os.path.join(data_dir_prefix,datafile))
        alldata['dataset_name'] = dataset_name
        return alldata

    elif data_split_method == 1:
        splitdatas = []
        datadir = '{}/{}/based_resid_split_data{}/fold_{}'.format(path_prefix,dataset_name,suffix,fold_idx)
        train = pd.read_csv(os.path.join(data_dir_prefix,datadir,'train.csv'))
        return train

    else:
        raise ValueError('split data method is valid')
###########

def get_splited_data(path_prefix,dataset_name,data_split_method,suffix = '',
         train_ratio=0.8,val_ratio=0.2,folder_num = 5,data_dir_prefix = ''):
    if data_split_method == 0:
        splitdatas = []
        assert train_ratio+val_ratio == 1
        datafile = '{}/{}/{}{}.csv'.format(path_prefix,dataset_name,dataset_name,suffix)
        alldata = pd.read_csv(os.path.join(data_dir_prefix,datafile))
        sample_len = len(alldata)
        for fold_idx in range(folder_num):
            alldata_shffled = shuffle(alldata,random_state=fold_idx)
            val_size    = math.floor(val_ratio * sample_len)
            val_dfs   = alldata_shffled[:val_size]
            wt_df = val_dfs.query("mutant=='WT'")
            val_dfs.drop(wt_df.index,inplace=True)
            val_dfs.reset_index(drop=True,inplace=True)
            train_dfs  = alldata_shffled[val_size:]

            train_final = pd.concat([alldata.iloc[:1],train_dfs])
            val_dfs = pd.concat([alldata.iloc[:1],val_dfs])

            train_final['dataset_name'] = dataset_name
            val_dfs['dataset_name'] = dataset_name
            splitdatas.append((train_final,val_dfs))

        return splitdatas

    elif data_split_method == 1:
        splitdatas = []

        for fold_idx in range(folder_num):
            datadir = '{}/{}/based_resid_split_data{}/fold_{}'.format(path_prefix,dataset_name,suffix,fold_idx)
            train = pd.read_csv(os.path.join(data_dir_prefix,datadir,'train.csv'))
            val = pd.read_csv(os.path.join(data_dir_prefix,datadir,'val.csv'))
            train['dataset_name'] = dataset_name
            val['dataset_name'] = dataset_name
            splitdatas.append((train,val))
        return splitdatas


###########
# dataset add 'msa_seq','coords_seq'

def get_mut_seq(data_df,wt_seq,column_name,offset):
    seq_list = []
    for i in range(len(data_df)):
        line = data_df.iloc[i]
        mutants = line['mutant']
        if mutants == 'WT':
            seq_list.append(wt_seq)
        else:
            seq_mut = wt_seq
            for mutant in mutants.split('-'):
                mut_idx = int(mutant[1:-1])-offset
                assert wt_seq[mut_idx] == mutant[0], ValueError('wild type seq is not consistent with mutant type')
                seq_mut = seq_mut[:mut_idx] + mutant[-1] + seq_mut[mut_idx+1:] #得到 mutant sequence 
            seq_list.append(seq_mut)
    data_df[column_name] = seq_list
    return data_df
