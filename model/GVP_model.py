import sys, time,os, random,copy
import collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleDict
from model.model_utils import GVP, GVPConvLayer, LayerNorm
from model.utils import *
# VEPmodel 
# RunGVP
class RunGVP(object):    
    def __init__(self,
            output_dir,
            dataset_names,
            device='cuda:0',
            node_in_dim = (6, 3),
            node_h_dim = (100,16),
            edge_in_dim = (32, 1),
            edge_h_dim =(32,1),
            lr = 1e-4,
            batch_size=128,
            top_k=15,data_category=False,
            multi_train=False,out_dim=1,
            n_ensembles=3,load_model_path = None,
            esm_msa_linear_hidden=128, num_layers=2,msa_in=True,
            drop_rate=0.1):

        if data_category:
            assert out_dim == 3
        else:
            assert out_dim ==1
        self.batch_size = batch_size
        self.dataset_names = dataset_names
        self.top_k = top_k
        self.data_category = data_category
        self.output_dir = output_dir
        self.device = device
        self.msa_in = msa_in
        CHARS = ["-", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                 "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

        self.letter_to_num = {c: i for i, c in enumerate(CHARS)}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
##################################
        
        if load_model_path:
            model_dict = torch.load(load_model_path,map_location=self.device)
            model = model_dict['model']
            model.load_state_dict(model_dict['model_para'])
            model.multi_train = False
            self.models = [model]

        else:
            self.models = [
            VEPModel(node_in_dim=node_in_dim, node_h_dim=node_h_dim, 
                     edge_in_dim=edge_in_dim, edge_h_dim=edge_h_dim,dataset_names=dataset_names,
                     multi_train = multi_train, esm_msa_linear_hidden = esm_msa_linear_hidden,
                     seq_in=True, num_layers=num_layers, drop_rate=drop_rate,
                     out_dim = out_dim,seq_esm_msa_in=msa_in).to(self.device) for _ in range(n_ensembles)]
            
        weight = torch.tensor([1,100],dtype=torch.float,device=self.device)
        self.Loss_c = nn.CrossEntropyLoss(weight = weight)
        self.Loss_mse = nn.MSELoss()
        self.batch_size = batch_size
        self.optimizers = [torch.optim.Adam(model.parameters(),lr=lr) for model in self.models]
        self._test_pack = None

##################################
    def Model_list(self):
        return self.models, self.optimizers
##################################

    def RunoneModel(self,model,optimizer,data_loader_dict_total,device,mode='train'):
        self.device = device
        losses_all = 0
        if mode == 'train':
            data_loader_dict = data_loader_dict_total[mode] # train_dataload get graph and wt_graph
            datasets_list = data_loader_dict.keys() # train_dataload list

            spearman_list = []
            count = 0
            target_list = []
            pred_list = []
            data_loader_dict_iter = {}
            for dataset in datasets_list:
                data_loader_dict_iter[dataset] = iter(data_loader_dict[dataset])
            while True:
                dataset = random.sample(datasets_list,1)[0]
                try:
                    (graph,wt_graph) = next(data_loader_dict_iter[dataset])
                except StopIteration:
                    break
    
                count +=1
        
                model.train()
                out = model(graph.to(self.device),wt_graph.to(self.device))
                target = graph.target.float().to(self.device)
    
                if self.data_category:
                    out_classfy,out_reg = out
                    target_category = torch.tensor(graph.target_category,dtype=torch.long,device=self.device) # category in TEM1_rescaled.csv
                    loss_classfy = self.Loss_c(out_classfy,target_category) # Loss_cross entropy in classfy
    
                else:
                    out_reg = out
                    loss_classfy = 0

                loss_reg = self.Loss_mse(out_reg,target) # 均方误差 nn.MSELoss()
                loss = loss_classfy + loss_reg
                loss.backward() # loss backward    
                losses_all += loss.item() # 返回标量的值  tensor的第一个词

                optimizer.step()
                optimizer.zero_grad()

                target_list.extend(target.cpu().detach().numpy()) # target list
                pred_list.extend(out_reg.cpu().detach().numpy()) # out_reg list

            pred_list = np.vstack(pred_list)[:,0]
            target_list = np.vstack(target_list)[:,0]
            spearman_v = spearman(pred_list,target_list) # 斯皮尔曼相关系数
        
            return losses_all/count,(pred_list,target_list),spearman_v

        elif mode == 'test' or 'val':
            
            data_loader_dict = data_loader_dict_total[mode]
            datasets_list = data_loader_dict.keys()

            count = 0
            losses_all = 0
            
            spearman_list = []
            outall_reg_all = []
            target_all_all = []
            mutant_all_list = []
            dataname_all_list = []

            for dataset in datasets_list:
                target_list = []
                pred_list = []
                mutant_list = []
                dataname_list = []
    
                for (graph,wt_graph) in data_loader_dict[dataset]:
                    with torch.no_grad():
                        model.eval()
                        count +=1
                        out = model(graph.to(device),wt_graph.to(device))
                        target = graph.target.float().to(device)
                        mutant = graph.mutant
                        #seq = cut([self.num_to_letter[int(a)] for a in graph.seq.to(device)],len(graph.mutant))
                        dataname = len(graph.mutant)*[dataset]
                        
                        if self.data_category:
                            out_classfy,out_reg = out
                            target_category = torch.tensor(graph.target_category,dtype=torch.long,device=self.device)
                            loss_classfy = self.Loss_c(out_classfy,target_category)
            
                        else:
                            out_reg = out
                            loss_classfy = 0
                        
                        target_list.extend(target.cpu().detach().numpy())
                        pred_list.extend(out_reg.cpu().detach().numpy())
                        mutant_list.extend(mutant)
                        dataname_list.extend(dataname)

                        loss_reg = self.Loss_mse(out_reg,target)
                        loss = loss_classfy + loss_reg
                        losses_all += loss.item()

                pred_list = np.vstack(pred_list)
                target_list = np.vstack(target_list)
                mutant_list = np.vstack(mutant_list)
                dataname_list = np.vstack(dataname_list)
                spearman_list.append(spearman(pred_list,target_list))    
                outall_reg_all.append(pred_list)
                target_all_all.append(target_list)
                mutant_all_list.append(mutant_list)
                dataname_all_list.append(dataname_list)
            
            return_out = (losses_all/count),(np.vstack(outall_reg_all)[:,0],np.vstack(target_all_all)[:,0]),spearman_list,np.vstack(mutant_all_list)[:,0],np.vstack(dataname_all_list)[:,0]

        return return_out

#####################################################

class VEPModel(nn.Module):
 
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim, dataset_names,
                 multi_train = False,
                 esm_msa_linear_hidden = 128,
                 seq_in=True, num_layers=2, drop_rate=0.1,seq_esm_msa_in=True,
                 out_dim = 3):
        
        super(VEPModel, self).__init__()
        self.node_h_dim = node_h_dim
        self.seq_esm_msa_in = seq_esm_msa_in
        self.out_dim = out_dim
        self.esm_msa_linear = nn.Linear(768,esm_msa_linear_hidden) # 768->128
        self.multi_train = multi_train

        if seq_esm_msa_in:
            node_in_dim = (node_in_dim[0] + esm_msa_linear_hidden, node_in_dim[1]) # [134,3]

        if seq_in:
            self.W_s = nn.Embedding(21, 20) # seq embedding 
            node_in_dim = (node_in_dim[0] + 20*3, node_in_dim[1]) # [194,3]
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None)) #[194,3] [100,16]
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)) # [32,1] [32,1]
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) # [100,16] [32,1]
            for _ in range(num_layers))

        ns, _ = node_h_dim # 100
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0))) # [100,16] -> [100,0]
            
        self.dense = nn.Sequential(
            nn.Linear(ns, ns//2), nn.ReLU(inplace=True), # [100] -> [50]
            nn.Dropout(p=drop_rate),
            nn.Linear(ns//2, ns*2) # [50] -> [100]
        )

        self.readout = nn.Sequential(
            AggregateLayer(d_model = ns*2), # 聚合 
            GlobalPredictor(d_model = ns*2, # 全局预测
                                d_h=128, d_out=out_dim)
        )
        if multi_train:
            readout_list = [copy.deepcopy(self.readout) for i in range(len(dataset_names))] #copy to other proteins 
            self.readout_dict = ModuleDict(dict(zip(dataset_names,readout_list))) 

    def forward(self,graph,wt_graph):
        out = self.forward1(graph,wt_graph) # out forward1
        out = self.dense(out) # linear trans
        if self.multi_train: # 多个蛋白合并
            out = self.readout_dict[graph.dataset_name[0]](out)
        else:
            out = self.readout(out)
            
        if self.out_dim ==3: # out 分类  out_classfy,out_reg
            return out[:,:2],out[:,2]
        elif self.out_dim ==1:
            return out[:,0]
        elif self.out_dim ==4:
            return out[:,:3],out[:,3]
        else:
            print('out dim not in [0,3], not implement')

    def forward1(self, graph,graph_wt):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        batch_num = graph_wt.batch[-1]+1
        h_V = (graph_wt.node_s,graph_wt.node_v) # h_v
        h_E = (graph_wt.edge_s,graph_wt.edge_v) # h_e
        seq = graph.seq # seq
        seq_wt = graph_wt.seq # wt_seq
        edge_index = graph_wt.edge_index # edge_index
        if seq is not None:
            seq = self.W_s(seq.long()) # embedding(seq)
            seq_wt = self.W_s(seq_wt.long()) # embedding(wt_seq)
            h_V = (torch.cat([h_V[0], seq,seq_wt,seq-seq_wt], dim=-1), h_V[1]) #[263,66] [263,3,3]

        if self.seq_esm_msa_in: #[h_V[0].shape = (bs*seqlen,dim)
            h_V = (torch.cat([h_V[0], self.esm_msa_linear(graph_wt.msa_rep[0])], dim=-1), h_V[1]) # [263,194] [263,3,3]

        h_V = self.W_v(h_V) #[194,3] [100,16]
        h_E = self.W_e(h_E) # [32,1] [32,1]
        for layerid,layer in enumerate(self.layers):
            h_V = layer(h_V, edge_index, h_E) # GVPConvLayer  # [100,16] [32,1]

        out = self.W_out(h_V) # [100,16] -> [100,0]
        hidden_dim = out.shape[-1]
        out = out.reshape(batch_num,-1,hidden_dim) # reshape output
        
        return out


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class AggregateLayer(nn.Module):
    def __init__(self, d_model=None, dropout=0.1):
        super(AggregateLayer, self).__init__()
        self.attn = nn.Sequential(collections.OrderedDict([
            ('layernorm', nn.LayerNorm(d_model)),
            ('fc', nn.Linear(d_model, 1, bias=False)),
            ('dropout', nn.Dropout(dropout)),
            ('softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, context):

        weight = self.attn(context) 
        output = torch.bmm(context.transpose(-1, -2), weight)
        output = output.squeeze(-1)
        return output


class GlobalPredictor(nn.Module):
    def __init__(self, d_model=None, d_h=None, d_out=None, dropout=0.5):
        super(GlobalPredictor, self).__init__()
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.predict_layer = nn.Sequential(collections.OrderedDict([
            # ('batchnorm', nn.BatchNorm1d(d_model)),
            ('fc1', nn.Linear(d_model, d_h)),
            ('tanh', nn.Tanh()),
            ('dropout', nn.Dropout(dropout)),
            ('fc2', nn.Linear(d_h, d_out))
        ]))

    def forward(self, x):
        if x.shape[0] !=1:
            x = self.batchnorm(x)
        x = self.predict_layer(x)
        return x

