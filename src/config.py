#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 16:00
# @Author  : Anonymous
# @Site    : 
# @File    : config.py
# @Software: PyCharm

# #Desc: config file

import warnings

class TrainConfig():
    """Config file"""
    model = 'Seagull'
    state = 'info'
    dataset = 'douban_mb'
    num_log = '2'
    checkpoint_dir = 'douban_mb'
    data_A = 'douban_movie'
    data_B = 'douban_book'
    loss_decimal_place = 4 

    learner = 'adam'
    num_m_step = 1  
    warm_up_step = 1 
    learning_rate = 0.0005 # mb0.001 (0.0007) mm 0.001 bm 0.002 
    epochs = 11
    eval_step = 1 
    eval_batch = 40960
    stopping_step = 10
    clip_grad_norm = None 
    valid_metric = ['hit', 'ndcg']
    valid_metric_bigger = "hit" 
    use_gpu = True
    device = 'cuda:0' # 'cuda:0'
    weight_decay = 5e-4

    batch_size = 4096
    embedding_size = 64
    n_layers = 2
    reg_weight = 1e-2
    ssl_temp = 0.1
    ssl_reg = 1e-7 # 
    threshold = 2 
    growing_factor = 1.25
    hyper_layers = 1 
    alpha = 1
    proto_reg = 8e-8 
    k = 100 
    topk = 10
    neg_num = 7

    overlap_ratio = None # overlap


    

def parse(self, kwargs):
    '''
    monkey parse
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))

# train
config = vars(TrainConfig)
config = {k:v for k,v in config.items() if not k.startswith('__')}


