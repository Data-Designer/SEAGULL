#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 16:00
# @Author  : Anonymous
# @Site    : 
# @File    : data.py
# @Software: PyCharm

# #Desc: dataset

import dgl
import torch
import operator
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch.nn.functional as F
from config import config
from utils import *
from torch.utils.data import DataLoader
from gensim.models.doc2vec import Doc2Vec




class Dataset():
    def __init__(self, fileName):
        self.fileName = fileName
        self.data, self.shape = self.getData()
        self.user_info, self.item_info = self.getFeature()
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getTrainDict()


    def getData(self):
        """dataset"""
        print("Loading %s data set..." % (self.fileName))
        data = []
        filePath = self.fileName + '/ratings_p.csv'
        u, i, maxr = 0, 0, 0.0
        file = pd.read_csv(filePath,index_col=0,header=0)
        file.columns = ["user_id","item_id","score"]
        file["user_id"] = file["user_id"].astype(int)
        file["item_id"] = file["item_id"].astype(int)
        file["score"] = file["score"].astype(float)
        
#         # for rule.py
#         ids = file.user_id.value_counts()
#         df = pd.DataFrame(columns=['ids'])
#         df["ids"] = ids
#         df = df[df['ids'] == 1]
#         delindexs = df.index
#         file = file[~file['user_id'].isin(delindexs)]
        
        u,i,maxr = file["user_id"].max(),file["item_id"].max(),file["score"].max() # 
#         u,i,maxr = 15577,file["item_id"].max(),file["score"].max() # amazonp 15577
        self.raw_data = file
        self.maxRate = maxr
        data = list(zip(file["user_id"].values.tolist(),file["item_id"].values.tolist(),file["score"].values.tolist(),[0]*file.shape[0]))

        print("Loading Success!\n"
              "Data Info:\n"
              "\tUser Num: {}\n"
              "\tItem Num: {}\n"
              "\tData Size: {}".format(u, i, len(data)))
        return data, [u, i]

    def getTrainTest(self):
        """split"""
        data = self.data
        data = sorted(data, key=lambda x: (x[0], x[3])) 
        train, test = [], []
        for i in range(len(data) - 1):
            user = data[i][0]-1  
            item = data[i][1]-1 
            rate = data[i][2]
            if data[i][0] != data[i + 1][0]:
                test.append((user, item, rate))
            else:
                train.append((user, item, rate)) 

        test.append((data[-1][0] - 1, data[-1][1] - 1, data[-1][2])) 
        return train, test

    def getTrainDict(self):
        """train inter"""
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getFeature(self):
        """doc feature"""
        feature = Doc2Vec.load(
            (self.fileName + "/Doc2vec_" + self.fileName.split("/")[-1] + "_VSize%02d" + ".model") % config['embedding_size']).docvecs.vectors_docs
        user_info, item_info = feature[:self.shape[0], :], feature[self.shape[0]:, :]
        return user_info, item_info

    def getEmbedding(self):
        """lookup embedding"""
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating # shape==[u，i]
        return np.array(train_matrix)
    
    def getSparseEmbedding(self):
        user,item,rating = [],[],[]
        for i in self.train:
            user.append(i[0])
            item.append(i[1])
            rating.append(i[2])
        train_matrix = sp.coo_matrix((rating,(user,item)),shape=(self.shape[0],self.shape[1]))
        return train_matrix
    

    def getInstance(self,data,negNum):
        """sample"""
        user,item,rate = [],[],[]
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict: 
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getPNInstance(self, data, negNum):
        """for bpr loss, rating"""
        user, pos_item, neg_items, rate = [], [], [], []
        for i in data:
            user.append(i[0])
            pos_item.append(i[1])
            rate.append(i[2])
            neg_item = []
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                neg_item.append(j)
            neg_items.append(neg_item)
        return np.array(user), np.array(pos_item), np.array(neg_items), np.array(rate)


    def getTestNeg(self,testData, negNum):
        """Test Negative sampling method"""
        user,item = [],[]
        for s in testData: # each user
            tmp_user,tmp_item = [], []
            u,i = s[0],s[1]
            tmp_user.append(u) #
            tmp_item.append(i)
            neglist = set()
            neglist.add(i)
            for t in range(negNum):
                j = np.random.randint(self.shape[1]) 
                while (u, j) in self.trainDict or j in neglist:
                    j = np.random.randint(self.shape[1])
                neglist.add(j)
                tmp_user.append(u)
                tmp_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        return [np.array(user), np.array(item)] # [[1,1,1...],[2,2,2...]] user*100


def get_douban_data():
    root = '/dfs/data/ORec/data/'
    dataName_A, data_A_path = config['data_A'], root+config['data_A']
    dataName_B, data_B_path = config['data_B'], root+config['data_B']

    dataset_A, dataset_B = Dataset(data_A_path), Dataset(data_B_path)

    # I re sort
    index_A = (pd.read_csv(data_A_path + '/' + dataName_A.split('_')[-1] + "_feature_p.csv")['UID'].values - 1).tolist()
    index_B = (pd.read_csv(data_B_path + '/' + dataName_B.split('_')[-1] + "_feature_p.csv")['UID'].values - 1).tolist()
    mapA_table = dict(zip(index_A, list(range(len(index_A)))))
    index_A_r = np.array(sorted(mapA_table.items(), key=operator.itemgetter(0)))[:, 1].tolist()
    mapB_table = dict(zip(index_B, list(range(len(index_B)))))
    index_B_r = np.array(sorted(mapB_table.items(), key=operator.itemgetter(0)))[:, 1].tolist()  
    dataset_A.item_info = np.take(dataset_A.item_info, index_A_r, axis=0)  
    dataset_B.item_info = np.take(dataset_B.item_info, index_B_r, axis=0)

    return dataset_A, dataset_B

def get_industry_data():
    root = '/dfs/data/ORec/data/'
    dataName_A, data_A_path = config['data_A'], root+config['data_A']
    dataName_B, data_B_path = config['data_B'], root+config['data_B']

    # inter
    dataset_A, dataset_B = Dataset(data_A_path), Dataset(data_B_path)
    return dataset_A, dataset_B


def get_amazon_data():
    root = '/dfs/data/ORec/data/'
    dataName_A, data_A_path = config['data_A'], root+config['data_A']
    dataName_B, data_B_path = config['data_B'], root+config['data_B']

    # inter
    dataset_A, dataset_B = Dataset(data_A_path), Dataset(data_B_path)
    return dataset_A, dataset_B


def train_test_split(dataset_A, dataset_B):
    train_A, testA = dataset_A.train, dataset_A.test
    train_B, testB = dataset_B.train, dataset_B.test
    testNegA, testNegB = dataset_A.getTestNeg(testA, 99), dataset_B.getTestNeg(testB, 99)

    train_u_A, train_i_A, train_r_A = dataset_A.getInstance(
        train_A, config['neg_num'])  # array,array,array Dataset *(1+negnum)
    train_len_A = len(train_u_A)
    shuffled_idx_A = np.random.permutation(np.arange(train_len_A))
    train_u_A = train_u_A[shuffled_idx_A]
    train_i_A = train_i_A[shuffled_idx_A]
    train_r_A = train_r_A[shuffled_idx_A]

    train_u_B, train_i_B, train_r_B = dataset_B.getInstance(
        train_B, config['neg_num'])
    train_len_B = len(train_u_B)
    shuffled_idx_B = np.random.permutation(np.arange(train_len_B))
    train_u_B = train_u_B[shuffled_idx_B]
    train_i_B = train_i_B[shuffled_idx_B]
    train_r_B = train_r_B[shuffled_idx_B]

    # overlap ratio
    dataA_overlap = []
    dataB_overlap = []
    if config['overlap_ratio']:
        overlap = list(set(train_u_A[train_r_A!=0]).intersection(set(train_u_B[train_r_B!=0]))) 
        overlap_mask = overlap[:round(len(overlap)*config['overlap_ratio'])] 
        train_u_A,train_i_A,train_r_A,train_len_A,dataA_overlap = mask_data(train_u_A,train_i_A,train_r_A,overlap_mask)
        train_u_B,train_i_B,train_r_B,train_len_B,dataB_overlap = mask_data(train_u_B,train_i_B,train_r_B,overlap_mask)

    return train_u_A, train_i_A, train_r_A ,train_u_B,train_i_B,train_r_B, testNegA, testNegB,\
           testA, testB, train_len_A, train_len_B, dataA_overlap,dataB_overlap


def train_test_split_bpr(dataset_A, dataset_B):
    train_A, testA = dataset_A.train, dataset_A.test
    train_B, testB = dataset_B.train, dataset_B.test
    testNegA, testNegB = dataset_A.getTestNeg(testA, 99), dataset_B.getTestNeg(testB, 99)

    train_u_A, train_i_A, train_neg_i_A, train_r_A = dataset_A.getPNInstance(
        train_A, config['neg_num'])  # array,array,array Dataset *(1+negnum),
    train_len_A = len(train_u_A)
    shuffled_idx_A = np.random.permutation(np.arange(train_len_A))
    train_u_A = train_u_A[shuffled_idx_A]
    train_i_A = train_i_A[shuffled_idx_A]
    train_neg_i_A = train_neg_i_A[shuffled_idx_A]
    train_r_A = train_r_A[shuffled_idx_A]

    train_u_B, train_i_B, train_neg_i_B, train_r_B = dataset_B.getPNInstance(
        train_B, config['neg_num'])
    train_len_B = len(train_u_B)
    shuffled_idx_B = np.random.permutation(np.arange(train_len_B))
    train_u_B = train_u_B[shuffled_idx_B]
    train_i_B = train_i_B[shuffled_idx_B]
    train_neg_i_B = train_neg_i_B[shuffled_idx_B]
    train_r_B = train_r_B[shuffled_idx_B]

    dataA_overlap = []
    dataB_overlap = []
    if config['overlap_ratio']:
        overlap = list(set(train_u_A[train_r_A!=0]).intersection(set(train_u_B[train_r_B!=0]))) 
        overlap_mask = overlap[:round(len(overlap)*config['overlap_ratio'])] # 
        train_u_A,train_i_A,train_neg_i_A,train_r_A,train_len_A,dataA_overlap = mask_data_bpr(train_u_A,train_i_A,train_neg_i_A,train_r_A,overlap_mask)
        train_u_B,train_i_B,train_neg_i_B,train_r_B,train_len_B,dataB_overlap = mask_data_bpr(train_u_B,train_i_B,train_neg_i_B,train_r_B,overlap_mask)

    return train_u_A, train_i_A, train_neg_i_A, train_r_A ,train_u_B,train_i_B,train_neg_i_B, train_r_B, testNegA, testNegB,\
           testA, testB, train_len_A, train_len_B, dataA_overlap, dataB_overlap



class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class UserItemBPRDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor,neg_item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.neg_item_tensor = neg_item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index],self.neg_item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


def normalize_ratings(ratings):
    """正则化"""
    max_rating = ratings.max()
    ratings['rating'] = ratings * 1.0 / max_rating
    return ratings



def create_dataset():
    """创建数据集"""
    if config['dataset'].split('_')[0]=='douban':
        dataset_A, dataset_B = get_douban_data()
    elif config['dataset'].split('_')[0]=='industry':
        dataset_A, dataset_B = get_industry_data()
    elif config['dataset'].split('_')[0]=='amazon':
        dataset_A, dataset_B = get_amazon_data()

    data_info = {"shape_A": dataset_A.shape, "shape_B": dataset_B.shape,
                 "maxRate_A": dataset_A.maxRate, "maxRate_B": dataset_B.maxRate}

    # train_test_split
    train_u_A, train_i_A, train_r_A, train_u_B, train_i_B, train_r_B, testNegA, testNegB, testA, testB, train_len_A, train_len_B, dataA_overlap, dataB_overlap = train_test_split(
        dataset_A, dataset_B)

    src, tgt, index_gap = get_graph_inter(dataset_A, dataset_B, testA, testB, dataA_overlap, dataB_overlap)
    data_uA_graph_fea, data_uB_graph_fea, data_iA_graph_fea, data_iB_graph_fea = get_graph_feature(dataset_A, dataset_B)
    graph_data = (src, tgt, index_gap, data_uA_graph_fea, data_uB_graph_fea, data_iA_graph_fea, data_iB_graph_fea)
    return data_info, dataset_A, dataset_B, train_u_A, train_i_A, train_r_A, train_u_B, train_i_B, train_r_B, \
           testNegA, testNegB, train_len_A, train_len_B, dataA_overlap, dataB_overlap, graph_data




def create_dataset_bpr():
    """dataset create"""
    if config['dataset'].split('_')[0]=='douban':
        dataset_A, dataset_B = get_douban_data()
    elif config['dataset'].split('_')[0]=='industry':
        dataset_A, dataset_B = get_industry_data()
    elif config['dataset'].split('_')[0]=='amazon':
        dataset_A, dataset_B = get_amazon_data()

    data_info = {"shape_A": dataset_A.shape, "shape_B": dataset_B.shape,
                 "maxRate_A": dataset_A.maxRate, "maxRate_B": dataset_B.maxRate}

    # train_test_split
    train_u_A, train_i_A,test_neg_i_A, train_r_A, train_u_B, train_i_B,test_neg_i_B, train_r_B, testNegA, testNegB, testA, testB, train_len_A, train_len_B, dataA_overlap, dataB_overlap = train_test_split_bpr(
        dataset_A, dataset_B)

    src, tgt, index_gap = get_graph_inter(dataset_A, dataset_B, testA, testB, dataA_overlap, dataB_overlap)
    data_uA_graph_fea, data_uB_graph_fea, data_iA_graph_fea, data_iB_graph_fea = get_graph_feature(dataset_A, dataset_B)
    graph_data = (src, tgt, index_gap, data_uA_graph_fea, data_uB_graph_fea, data_iA_graph_fea, data_iB_graph_fea)
    return data_info, dataset_A, dataset_B, train_u_A, train_i_A,test_neg_i_A, train_r_A, train_u_B, train_i_B,test_neg_i_B, train_r_B, \
           testNegA, testNegB, train_len_A, train_len_B, dataA_overlap, dataB_overlap, graph_data


def create_dataloader(data_info,batch_size):
    users,items,ratings = data_info
    dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users).to(config['device']),
                                    item_tensor=torch.LongTensor(items).to(config['device']),
                                    target_tensor=torch.FloatTensor(ratings).to(config['device']))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False) 


def create_dataloader_bpr(data_info,batch_size):
    users,items,neg_items,ratings = data_info
    dataset = UserItemBPRDataset(user_tensor=torch.LongTensor(users).to(config['device']),
                                    item_tensor=torch.LongTensor(items).to(config['device']),
                                  neg_item_tensor=torch.LongTensor(neg_items).to(config['device']),
                                    target_tensor=torch.FloatTensor(ratings).to(config['device']))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False) 



