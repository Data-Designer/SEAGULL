#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 16:00
# @Author  : Anonymous
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

# #Desc: utils tools
import dgl
import time
import logging
import colorlog
import os
import re
import math
import torch
import datetime
import random
import pandas as pd
import numpy as np
import torch.nn as nn
from models.ngl import *
from config import *
from colorama import init
from collections import defaultdict
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from functools import wraps
from torch.nn.init import xavier_uniform_, constant_




def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def print_info(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_time = end_time - start_time
        print("execute time running %s: %s seconds" % (func.__name__, duration_time))
        return result

    return wrapper


def graph_reid(dataset_A, dataset_B):
    """read"""
    src = dataset_A.raw_data
    tgt = dataset_B.raw_data

    src['user_id'] = src['user_id']-1
    tgt['user_id'] = tgt['user_id']-1
    src['item_id'] = src['item_id']-1
    src_item_max = max(src['item_id'])
    tgt['item_id'] = src_item_max + tgt['item_id']

    return src, tgt, src_item_max

def get_graph_inter(dataset_A, dataset_B, testA, testB,dataA_overlap, dataB_overlap):
    src, tgt, index_gap = graph_reid(dataset_A, dataset_B)  # ID_REID
    # mask
    src, tgt = mask_edge(testA, testB, src, tgt, index_gap)
    # mask overlap
    if config['overlap_ratio']:
        src, tgt = mask_edge(dataA_overlap, dataB_overlap, src, tgt, index_gap)
    return src, tgt, index_gap


def get_graph_feature(dataset_A, dataset_B):
    # data_uinfo = np.maximum(dataset_A.user_info, dataset_B.user_info)
    # data_uinfo = torch.from_numpy(data_uinfo)
    data_uA_graph_fea = torch.from_numpy(dataset_A.user_info).to(config['device'])
    data_uB_graph_fea = torch.from_numpy(dataset_B.user_info).to(config['device'])
    data_iA_graph_fea = torch.from_numpy(dataset_A.item_info).to(config['device'])
    data_iB_graph_fea = torch.from_numpy(dataset_B.item_info).to(config['device'])
    return data_uA_graph_fea, data_uB_graph_fea, data_iA_graph_fea, data_iB_graph_fea




def dense_to_sparse(matrix):
    coo_a = matrix.to_sparse()
    neighbor = torch.sparse_coo_tensor(coo_a._indices(),torch.ones_like(coo_a._values(), device=matrix.device), matrix.shape)
    return coo_a, neighbor

def aggravate_domain(src,tgt,u_feature):
    graph_src = dgl.graph((src[0,:],src[1,:])).to(config['device'])
    graph_tgt = dgl.graph((tgt[0,:],tgt[1,:])).to(config['device'])
    graph_src.ndata['info'] = u_feature
    graph_tgt.ndata['info'] = u_feature
    net = GraphNet().to(config['device'])
    u_feature_src = net(graph_src, u_feature)
    u_feature_tgt = net(graph_tgt, u_feature)
    return u_feature_src,u_feature_tgt

def aggravate_domain_matrix(src,tgt,u_feature):
#     u_feature,_ = dense_to_sparse(u_feature)
    u_feature_src = torch.mm(src,u_feature) # D,d
    u_feature_tgt = torch.mm(tgt,u_feature)

#     u_feature_src = torch.sparse.mm(src, u_feature)
#     u_feature_tgt = torch.sparse.mm(tgt, u_feature)

    return u_feature_src,u_feature_tgt



def get_time(f):
    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

def einsum_matmul(a,b):
    """speed test"""
    c = torch.einsum('ij,jk->ik', a, b)
    return c


def mask_data(train_u,train_i,train_r,overlap_mask):
    """mask data"""
    data = pd.DataFrame()
    data['user_id'], data['item_id'], data['score'] = train_u,train_i,train_r
    data = data.loc[~data['user_id'].isin(overlap_mask)]
    data_overlap = data.loc[data['user_id'].isin(overlap_mask)] # filter
    train_len = data.shape[0]
    return data['user_id'].values, data['item_id'].values, data['score'].values,train_len,data_overlap


def mask_data_bpr(train_u,train_i,train_neg_i,train_r,overlap_mask):
    """mask data"""
    data = pd.DataFrame()
    neg_col = ['neg'+str(i) for i in range(config['neg_num'])]
    data['user_id'], data['item_id'], data[neg_col], data['score'] = train_u,train_i,train_neg_i,train_r # str
    data = data.loc[~data['user_id'].isin(overlap_mask)]
    data_overlap = data.loc[data['user_id'].isin(overlap_mask)] # filter
    train_len = data.shape[0]
    return data['user_id'].values, data['item_id'].values,data[neg_col].values, data['score'].values, train_len, data_overlap


def mask_edge(testA, testB, src, tgt, index_gap):
    """diff"""
    pairs_A = np.array(testA)[:, :2].astype('int')  
    pairs_B = np.array(testB)[:, :2].astype('int')  
    pairs_B[:, 1] = index_gap + pairs_B[:, 1]
    pairs_A = pd.DataFrame({'user_id': pairs_A[:, 0], 'item_id': pairs_A[:,1],'score':0})
    pairs_B = pd.DataFrame({'user_id': pairs_B[:, 0], 'item_id': pairs_B[:, 1],'score':0})

    src = set_diff_df(src,pairs_A)
    tgt = set_diff_df(tgt,pairs_B)
    return src, tgt


def set_diff_df(A,B):
    """diff"""
    A = pd.concat((A,B),axis=0,ignore_index=True)
    A = pd.concat((A,B),axis=0,ignore_index=True)
    A = A.drop_duplicates(subset=['user_id','item_id'],keep=False).reset_index(drop=True)
    return A


def get_local_time():
    r"""Get current time"""
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033[' 
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class RemoveColorFilter(logging.Filter):
    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True


log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}

def init_logger(config,root=None):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    # log dir
    init(autoreset=True)
    if root is None:
        LOGROOT = '/dfs/data/ORec/log/' + config['dataset'] + '/log' + config['num_log'] + '/'
    else:
        LOGROOT = root
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)

    logfilename = '{}-{}.log'.format(config['model'], get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    # log level
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])

def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping"""
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string."""

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)


def dict2str(result_dict):
    r""" convert result dict to str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ' : ' + str(value) + '    '
    return result_str


def get_tensorboard(logger):
    r""" Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for
    """
    base_path = '../log/' + config['dataset'] + '/log' + config['num_log'] + '/run'


    dir_name = None
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            dir_name = os.path.basename(getattr(handler, 'baseFilename')).split('.')[0]
            break
    if dir_name is None:
        dir_name = '{}-{}'.format('model', get_local_time())

    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer


def calculate_valid_score(valid_result, valid_metric=None):
    r""" return valid score from valid result"""
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result['hit']



def getHIT(ranklist, targetItem):
    for item in ranklist:
        if item == targetItem:
            return 1
    return 0


def getNDCG(ranklist, targetItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == targetItem:
            return math.log(2) / math.log(i + 2)
    return 0


def reg(tensor):
    return torch.sqrt(torch.sum(torch.square(tensor), axis=1) + 1e-8)


def l2_loss(tensor):
    loss = torch.sum(tensor.square())
    return loss

def calculate_metrics(scores, batched_data):
    """metric,B,100,1"""
    funcs = {'hit':getHIT, 'ndcg': getNDCG}
    batched_data = batched_data[1].cpu().numpy() # predict
    targets = batched_data[:, 0]
    scores = scores.cpu().numpy()
    result = defaultdict(list)
    for metric in config['valid_metric']:
        func = funcs[metric]
        ranklists = np.take_along_axis(batched_data, np.argsort(-scores), axis=-1)[:, :config['topk']] 
        for target, ranklist in zip(targets, ranklists):
            tmp = func(ranklist, target)
            result[metric].append(tmp)
    return result

def calculate_metrics_full(scores, batched_data, max_item_num):
    """metric,B,100,1"""
    funcs = {'hit':getHIT, 'ndcg': getNDCG}
    batched_data = batched_data[1].cpu().numpy() # predict
    targets = batched_data[:, 0]
    item = np.arange(max_item_num)
    all_items = np.tile(item,(batched_data.shape[0],1))
    scores = scores.cpu().numpy()
    result = defaultdict(list)
    for metric in config['valid_metric']:
        func = funcs[metric]
        ranklists = np.take_along_axis(all_items, np.argsort(-scores), axis=-1)[:, :config['topk']] 
        for target, ranklist in zip(targets, ranklists):
            tmp = func(ranklist, target)
            result[metric].append(tmp)
    return result





def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False



def xavier_uniform_initialization(module):
    r""" initialization"""
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking"""
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

def criterion(pred, label): # BCE loss
    pred = torch.clamp(pred, min=1e-7, max=1-1e-7) # avoid nan
    loss = label * torch.log(pred) + (1 - label) * torch.log(1 - pred)
    return loss



class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings, L2"""
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss

class SPLoss(nn.Module):
    def __init__(self, n_samples):
        super(SPLoss, self).__init__()
        self.threshold = config['threshold'] # adjust by mean emprical loss
        self.growing_factor = config['growing_factor']
        self.v = torch.zeros(n_samples).int().to(config["device"])

    def forward(self, super_loss, index):
        v = self.spl_loss(super_loss * 1e-7) 
        self.v[index] = v
        return (super_loss * v).sum()

    def increase_threshold(self):
        self.threshold *= self.growing_factor

    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.int()



if __name__ == '__main__':
    scores = torch.Tensor(3,3)
    item = torch.LongTensor(3,3)
    print(scores,item)