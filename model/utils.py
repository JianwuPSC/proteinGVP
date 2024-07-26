import sys
import torch
import os 
import pandas as pd
import math
from sklearn.utils import shuffle
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, ndcg_score

# sperman, ndcg, aucroc

def cut(obj, sec):
    return [obj[i:i+sec] for i in range(0,len(obj),sec)]

def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        print('pred value is almost same,var is {}'.format(np.var(y_pred)))
        return 0.0
    return spearmanr(y_pred, y_true).correlation

def ndcg_old(y_pred, y_true):
    y_true_normalized = (y_true - y_true.mean()) / (y_true.std()+0.0000001)
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))

def ndcg(y_pred, y_true):
    min_ytrue = np.min(y_true)
    if min_ytrue <0:
        y_true = y_true + abs(min_ytrue)
    k = math.floor(len(y_true)*0.01)
    return ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1),k=k)

def aucroc(y_pred, y_true, y_cutoff):
    y_true_bin = (y_true >= y_cutoff)
    return roc_auc_score(y_true_bin, y_pred, average='micro')

def t_sne(data,components=2):
    tsne = TSNE(n_components=2, random_state=0)
    embedding = tsne.fit_transform(data)
    return embedding[:,0],embedding[:,1]

def gradient_cosine_similarity(gradient1,gradient2):
    dot_product = np.dot(gradient1, gradient2)
    # 计算梯度模长
    norm1 = np.linalg.norm(gradient1)
    norm2 = np.linalg.norm(gradient2)
    # 计算余弦相似性
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity

class Logger(object):
    """Writes both to file and terminal"""
    def __init__(self, savepath, mode='a'):
        self.terminal = sys.stdout
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.log = open(os.path.join(savepath, 'logfile.log'), mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def randomSeed(random_seed):
    """Given a random seed, this will help reproduce results across runs"""
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
