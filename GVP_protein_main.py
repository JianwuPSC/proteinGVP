import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import multiprocessing

import os 
import sys
import pandas as pd
import numpy as np

sys.path.append('/home/wuj/data/protein_design/GVP_protein')
from model.Dataset import GVPdataset
from model.Dataset import Dataset_in
from model.GVP_model import RunGVP
from model.utils import *

#cuda_condition = torch.cuda.is_available() and with_cuda
#device = torch.device("cuda:0" if cuda_condition else "cpu")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#conda activate esmfold
#python GVP_protein_main.py --path_prefix=/home/wuj/data/protein_design/GVP_protein/params/directed_evolution_input_all_datasets --sample_names 'HSP90' --dataset_config=/home/wuj/data/protein_design/GVP_protein/params/data_config.yaml --output_dir=sample_HSP90 --epochs=100 --multi_model=False --high_order_train=False --test_names None
#python GVP_protein_main.py --path_prefix=/home/wuj/data/protein_design/GVP_protein/params/directed_evolution_input_all_datasets --sample_names 'HSP90' 'TEM1' --dataset_config=/home/wuj/data/protein_design/GVP_protein/params/data_config.yaml --output_dir=sample_HSP90 --epochs=5 --multi_model=True --high_order_train=False --test_names=HSP90
#python GVP_protein_main.py --path_prefix=/home/wuj/data/protein_design/GVP_protein/params/directed_evolution_input_all_datasets --sample_names 'FOS_JUN' 'AVGFP' --dataset_config=/home/wuj/data/protein_design/GVP_protein/params/data_config.yaml --output_dir=sample_HSP90 --epochs=100 --multi_model=True --high_order_train=True --test_names=TEM1

def get_args_parser():
    parser = argparse.ArgumentParser('GVP-MSA-GNN pre-training', add_help=True)

    parser.add_argument("--path_prefix", required=True, type=str, help="pathway in train dataset")
    parser.add_argument("--sample_names", required=True, nargs='+', action='store', type=str, help="train sample namses")
    parser.add_argument("--test_names", default=None, type=str, help="test sample namses")
    parser.add_argument("--dataset_config", required=True, type=str, help="dataset config")
    parser.add_argument("--mut_prefix", type=str, default=None, help="mutant prefix (recale)")
    
    parser.add_argument("--output_dir", required=True, type=str, help="ex output/model and output")
    parser.add_argument("--load_model_path", type=str, default=None, help="load model path")
    parser.add_argument("--multi_model", required=True, type=bool, default=False, help="Multi model train")
    parser.add_argument("--high_order_train", default=False, type=bool, help="high_order_train train or not")
    parser.add_argument("--high_order_input", default=None, type=str, help="high_order_input signal mutant")

    parser.add_argument("--top_k", type=int, default=15, help="hidden size of transformer model")
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--device", type=str, default='cuda:0', help="training with cuda:0")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=0, help="CUDA device ids")

    parser.add_argument("--save_checkpoint", type=bool, default=True, help="Save checkpoint: true or false")
    parser.add_argument("--save_prediction", type=bool, default=True, help="Save prediction: true or false")
    parser.add_argument("--data_category", type=bool, default=False, help="Data category: true or false")
    parser.add_argument("--msa_in", type=bool, default=True, help="Msa_in: true or false")
    parser.add_argument("--n_ensembles", type=int, default=3, help="Number of models in ensemble")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--num_workers", type=int, default=1, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    return parser


def main (args, mode = 'train'):

    if mode == 'train':

        print(f"Loading Train Dataset {args.sample_names}")
        train_data_dict={}
        val_data_dict={}
        test_data_dict={}

        if args.high_order_train :
            print(f'signal mutant -> high order mutant {args.test_names}')
            for data_name in args.sample_names:
                dataset = Dataset_in(args.path_prefix,data_name,args.dataset_config,'_all',folder_num=0,mode='train')
                train_dataset,val_dataset = dataset.get_graphset()
                train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
                train_data_dict[data_name] = train_data_loader
                val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False)
                val_data_dict[data_name] = val_data_loader
            # test signal mutant
            if args.test_names is not None:
                dataset = Dataset_in(args.path_prefix,args.test_names,args.dataset_config,'_single',folder_num=0,mode='train')
                train_dataset,val_dataset = dataset.get_graphset()
                train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
                train_data_dict[args.test_names] = train_data_loader
                val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False)
                val_data_dict[args.test_names] = val_data_loader

                dataset = Dataset_in(args.path_prefix,args.test_names,args.dataset_config,'_muti',folder_num=0,mode='test')
                test_dataset = dataset.get_graphset()
                test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
                test_data_dict[args.test_names] = test_data_loader
                print(f'{next(iter(test_data_loader))[0].dataset_name[0:10]}  {next(iter(test_data_loader))[0].mutant[0:10]}')

        else:
            if args.multi_model :
                for data_name in args.sample_names:
                    dataset = Dataset_in(args.path_prefix,data_name,args.dataset_config,args.mut_prefix,folder_num=0,mode='train')
                    train_dataset,val_dataset = dataset.get_graphset()
                    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
                    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False)
                    train_data_dict[data_name] = train_data_loader
                    val_data_dict[data_name] = val_data_loader
            else :
                for data_name in list([args.sample_names]):
                    dataset = Dataset_in(args.path_prefix,data_name,args.dataset_config,args.mut_prefix,folder_num=0,mode='train')
                    train_dataset,val_dataset = dataset.get_graphset()
                    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
                    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False)
                    train_data_dict[data_name] = train_data_loader
                    val_data_dict[data_name] = val_data_loader

            if args.test_names is not None:
                dataset = Dataset_in(args.path_prefix,args.test_names,args.dataset_config,args.mut_prefix,folder_num=0,mode='test')
                test_dataset = dataset.get_graphset()
                test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
                test_data_dict[args.test_names] = test_data_loader


        print(f"Creating Dataloader  train:{train_data_dict.keys()}  val:{val_data_dict.keys()}  test:{test_data_dict.keys()}")
        dataset_load = {'train':train_data_dict, 'val':val_data_dict, 'test':test_data_dict}

        print("Modeling")
        val_pred_ensemble = 0
        val_target_ensemble = 0
        test_pred_ensemble = 0
        test_target_ensemble = 0

        if args.multi_model :
            rungvp = RunGVP(
                    output_dir=os.path.join(args.output_dir,'~'.join(args.sample_names)),
                    dataset_names=train_data_dict.keys(),
                    device = args.device,
                    load_model_path = args.load_model_path,
                    data_category=args.data_category,
                    lr = args.lr,
                    batch_size = args.batch_size,
                    n_ensembles=args.n_ensembles,
                    multi_train=args.multi_model,
                    msa_in=args.msa_in,)
        else:
            rungvp = RunGVP(
                    output_dir=os.path.join(args.output_dir,'{}'.format(args.sample_names)),
                    dataset_names=[args.sample_names],
                    device = args.device,
                    load_model_path = args.load_model_path,
                    data_category=args.data_category,
                    lr = args.lr,
                    batch_size = args.batch_size,
                    n_ensembles=args.n_ensembles,
                    multi_train=args.multi_model,
                    msa_in=args.msa_in,)

        if args.test_names is not None :
            rungvp_test = RunGVP(
                    output_dir=os.path.join(args.output_dir,'{}'.format(args.sample_names)),
                    dataset_names=test_data_dict.keys(),
                    device = args.device,
                    load_model_path = args.load_model_path,
                    data_category=args.data_category,
                    lr = args.lr,
                    batch_size = args.batch_size,
                    n_ensembles=args.n_ensembles,
                    multi_train=False,
                    msa_in=args.msa_in,) 

        logger = Logger(args.output_dir)
        models, optimizers = rungvp.Model_list()
        names_file = '_'.join(args.sample_names)

        for midx, (model, optimizer) in enumerate(zip(models,optimizers)):
            for epoch in range(args.epochs):

                # Training
                train_losses_all,(train_pred_list,train_target_list),spearman_v_train = rungvp.RunoneModel(model,optimizer,dataset_load,device='cuda:0',mode='train')
                logger.write('Epoch{},train total loss: {},  Spearman:{}\n'.format(epoch,train_losses_all,spearman_v_train))
    
                # Validation
                val_losses,(val_pred,val_target),spearman_v_val,val_mutant,val_name = rungvp.RunoneModel(model,optimizer,dataset_load,device='cuda:0',mode='val')
                logger.write('Epoch{},val total loss: {},  Spearman:{}\n'.format(epoch,val_losses,spearman_v_val))
                val_best_pred_target = (val_pred,val_target,val_mutant,val_name)

                # Testing
                if args.test_names is not None :
                    test_losses,(test_pred,test_target),spearman_v_test,test_mutant,test_name = rungvp_test.RunoneModel(model,optimizer,dataset_load,device='cuda:0',mode='test')
                    logger.write('Epoch{},test total loss: {},  Spearman:{}\n'.format(epoch,test_losses,spearman_v_test))
                    test_best_pred_target = (test_pred,test_target,test_mutant,test_name)

            val_pred_ensemble += np.array(val_best_pred_target[0])
            val_target_ensemble += np.array(val_best_pred_target[1])

            if args.test_names is not None:
                test_pred_ensemble += np.array(test_best_pred_target[0])
                test_target_ensemble += np.array(test_best_pred_target[1])
            
            best_model_para = model.state_dict()
            spearman_v_val_mean = sum(spearman_v_val)/len(spearman_v_val)

            if args.save_checkpoint:
                best_stat = {'model_para':best_model_para,
                'model':model,
                'epoch':epoch,'pred_target':val_best_pred_target,
                'best_val_metrics':spearman_v_val_mean,}
                torch.save(best_stat, os.path.join(args.output_dir,'{}_epoch{}_ensemble{}.pt'.format(names_file,epoch,midx)))

            if args.save_prediction:
                dataframe = pd.DataFrame({'pred':val_pred_ensemble,'target':val_target_ensemble,'mutant':val_best_pred_target[2],'name':val_best_pred_target[3]})
                dataframe.to_csv(os.path.join(args.output_dir,'{}_val_pred_fold{}.csv'.format(names_file,midx)))
                if args.test_names is not None:
                    dataframe = pd.DataFrame({'pred':test_pred_ensemble,'target':test_target_ensemble,'mutant':test_best_pred_target[2],'name':test_best_pred_target[3]})
                    dataframe.to_csv(os.path.join(args.output_dir,'{}_test_pred_fold{}.csv'.format(args.test_names,midx)))

            ensemble_metrics_spearman = spearman(val_pred_ensemble,val_target_ensemble)
            ensemble_metrics_ndcg = ndcg(val_pred_ensemble,val_target_ensemble)
            logger.write('fold {}, spearman is {}, ndcg is {}\n'.format(
                         midx,ensemble_metrics_spearman,ensemble_metrics_ndcg))

            print(f'Finished {midx} training processes')
    ###################################################################################################################################           

    else:  #test

        data_name = args.test_names
        test_dataset = GVPdataset(args.path_prefix, data_name, args.dataset_config, args.mut_prefix,folder_num=0,mode='test')
        test_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False)
        test_data_dict[data_name] = test_data_loader
        dataset_load = {'test':test_data_dict}

        test_pred_ensemble = 0
        test_target_ensemble = 0
        test_mutant_ensemble=[]
        test_name_ensemble=[]

        rungvp = RunGVP(
                output_dir=os.path.join(args.output_dir,'{}'.format(args.sample_names)),
                dataset_names=[args.sample_names],
                device = args.device,
                load_model_path = args.load_model_path,
                data_category=args.data_category,
                lr = args.lr,
                batch_size = args.batch_size,
                n_ensembles=args.n_ensembles,
                multi_train=args.multi_model,
                msa_in=args.msa_in,)

        logger = Logger(args.output_dir)

        test_losses,(test_pred,test_target),spearman_v_test,test_mutant,test_name = rungvp.RunoneModel(model,optimizer,dataset_load,device='cuda:0',mode='test')
        logger.write('Epoch{},test total loss: {},  Spearman:{}\n'.format(epoch,test_losses,spearman_v_test))
        test_best_pred_target = (test_pred,test_target,test_mutant,test_name)

        test_pred_ensemble += np.array(test_best_pred_target[0])
        test_target_ensemble += np.array(test_best_pred_target[1])
        test_mutant_ensemble += np.array(test_best_pred_target[2])
        test_name_ensemble += np.array(test_best_pred_target[3])

        spearman_v_test_mean = sum(spearman_v_test)/len(spearman_v_test)

        if args.save_prediction:
            dataframe = pd.DataFrame({'pred':test_pred_ensemble,'target':test_target_ensemble,'mutant':test_mutant_ensemble,'name':test_name_ensemble})
            dataframe.to_csv(os.path.join(args.output_dir,'{}_test_pred_fold{}.csv'.format(names_file,midx)))

        ensemble_metrics_spearman = spearman(test_pred_ensemble,test_target_ensemble)
        ensemble_metrics_ndcg = ndcg(test_pred_ensemble,test_target_ensemble)
        logger.write('fold {}, spearman is {}, ndcg is {}\n'.format(
                      midx,ensemble_metrics_spearman,ensemble_metrics_ndcg))

        times.sleep(0.5)


if __name__ =='__main__':

    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    #multiprocessing.set_start_method('spawn')
    mode = 'train' # train/evalu

    if mode == 'train':
        print('========  Train  =============================================================')
        main(args, mode=mode)
    else:
        print('========  Test  ==============================================================')
        main(args, mode=mode)
