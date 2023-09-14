#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/22 15:45
# @Author  : Anonymous
# @Site    : 
# @File    : main.py
# @Software: PyCharm

# #Desc: main
import pandas as pd
from data import *
from logging import getLogger
from config import *
from utils import *
from trainer import *
from models.seagull import *


def run_single_model(config):
    # logger initial
    print("log initial")
    init_logger(config)
    init_seed(seed=42, reproducibility=True)
    logger = getLogger()
    logger.info(config)

    # create dataset
    data_info, dataset_A, dataset_B, train_u_A, train_i_A, train_r_A, train_u_B, train_i_B, train_r_B, \
    testNegA, testNegB, train_len_A, train_len_B, dataA_overlap, dataB_overlap, graph_data = create_dataset()
    logger.info(data_info)

    A_info = [train_u_A, train_i_A, train_r_A]
    B_info = [train_u_B, train_i_B, train_r_B]
    dataA_trainloader = create_dataloader(A_info, config["batch_size"])
    dataB_trainloader = create_dataloader(B_info, config["batch_size"])

    A_test = [testNegA[0], testNegA[1], np.zeros_like(testNegA[0])] # Test_user，[Test_item_100]
    B_test = [testNegB[0], testNegB[1], np.zeros_like(testNegB[0])]
    dataA_testloader = create_dataloader(A_test, config["eval_batch"])
    dataB_testloader = create_dataloader(B_test, config["eval_batch"])

    user_dict = {}
    overlap_user = list(set(train_u_A[train_r_A != 0]).intersection(set(train_u_B[train_r_B != 0])))
    A_distinct = list(set(train_u_A[train_r_A != 0]) - set(overlap_user))
    B_distinct = list(set(train_u_B[train_r_B != 0]) - set(overlap_user))

    # model loading & initialization
    user_dict["overlap"], user_dict["dataset_A"], user_dict["dataset_B"] = overlap_user, A_distinct, B_distinct
    model = Seagull(config, (dataset_A, dataset_B), graph_data, user_dict)
    logger.info(model)
    
    config['max_item_size_A'] = dataset_A.shape[1]
    config['max_item_size_B'] = dataset_B.shape[1] 

    # model trainer
    trainer = SeagullTrainer(config, model)
    logger.info("init success!")

    # model training
    best_valid_result = trainer.fit(train_data=(dataA_trainloader, dataB_trainloader),
                                                      valid_data=(dataA_testloader, dataB_testloader))

    # model evaluation
    test_result = trainer.evaluate(eval_data=(dataA_testloader, dataB_testloader)) 
    
    # save 
    root = '/dfs/data/ORec/log/'+config['dataset'] + '/log'+ config['num_log'] + '/'
    pd.DataFrame(model.user_embeddings_src.weight.data.cpu().numpy()).to_csv(root+'src_user1.csv')
    pd.DataFrame(model.user_embeddings_tgt.weight.data.cpu().numpy()).to_csv(root+'tgt_user1.csv')


    # log info
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')




def run_bpr_model(config):
    print("log initial")
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # create dataset
    data_info, dataset_A, dataset_B, train_u_A, train_i_A, train_neg_i_A, train_r_A, train_u_B, train_i_B, train_neg_i_B, train_r_B, \
    testNegA, testNegB, train_len_A, train_len_B, dataA_overlap, dataB_overlap, graph_data = create_dataset_bpr()
    logger.info(data_info)

    A_info = [train_u_A, train_i_A, train_neg_i_A, train_r_A]
    B_info = [train_u_B, train_i_B, train_neg_i_B, train_r_B]
    dataA_trainloader = create_dataloader_bpr(A_info, config["batch_size"])
    dataB_trainloader = create_dataloader_bpr(B_info, config["batch_size"])

    A_test = [testNegA[0], testNegA[1], np.zeros_like(testNegA[0])] # Test_user，[Test_item_100]
    B_test = [testNegB[0], testNegB[1], np.zeros_like(testNegB[0])]
    dataA_testloader = create_dataloader(A_test, config["eval_batch"])
    dataB_testloader = create_dataloader(B_test, config["eval_batch"])

    user_dict = {}
    overlap_user = list(set(train_u_A[train_r_A != 0]).intersection(set(train_u_B[train_r_B != 0])))
    A_distinct = list(set(train_u_A[train_r_A != 0]) - set(overlap_user))
    B_distinct = list(set(train_u_B[train_r_B != 0]) - set(overlap_user))

    # model loading & initialization
    user_dict["overlap"], user_dict["dataset_A"], user_dict["dataset_B"] = overlap_user, A_distinct, B_distinct
    model = Seagull(config, (dataset_A, dataset_B), graph_data, user_dict)
    logger.info(model)
    
    config['max_item_size_A'] = dataset_A.shape[1]
    config['max_item_size_B'] = dataset_B.shape[1] 

    # model trainer
    trainer = SeagullTrainer(config, model)
    logger.info("init success!")


    # model training
    best_valid_result = trainer.fit(train_data=(dataA_trainloader, dataB_trainloader),
                                    valid_data=(dataA_testloader, dataB_testloader))


    # model evaluation
    test_result = trainer.evaluate(eval_data=(dataA_testloader, dataB_testloader)) 
    


    # log info
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')



if __name__ == '__main__':
    # define config
    seagull_config = config
    # run single model
    run_single_model(seagull_config)
    # run bpr model
#     run_bpr_model(seagull_config)