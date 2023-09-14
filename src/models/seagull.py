#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/1/21 14:14
# @Author  : Anonymous
# @Site    : 
# @File    : seagull.py
# @Software: PyCharm

# #Desc: algorithm imp
import torch
import faiss
import recbole
import time
import numpy as np

import torch.nn as nn
from utils import *
from models.ngl import *



class Seagull(nn.Module):
    def __init__(self, config, dataset, graph_data, user_dict):
        
        super(Seagull, self).__init__()
        self.device = config['device']


        # load dataset info
        self.config = config
        self.dataset_A = dataset[0]
        self.dataset_B = dataset[1]
        self.src, self.tgt, self.index_gap, self.data_uA_graph_fea, self.data_uB_graph_fea, self.data_iA_graph_fea, self.data_iB_graph_fea = graph_data
        self.overlap, self.unique_A, self.unique_B = user_dict["overlap"], user_dict["dataset_A"], user_dict["dataset_B"]


        self.n_users_src = self.dataset_A.shape[0]
        self.n_items_src = self.dataset_A.shape[1]
        self.n_users_tgt = self.dataset_B.shape[0]
        self.n_items_tgt = self.dataset_B.shape[1]


        # load params info
        self.latent_dim = self.config['embedding_size']
        self.n_layers = self.config['n_layers']
        self.reg_weight = self.config['reg_weight']

        self.ssl_temp = self.config['ssl_temp']
        self.ssl_reg = self.config['ssl_reg']
        self.hyper_layers = self.config['hyper_layers']

        self.alpha = self.config['alpha']

        self.proto_reg = self.config['proto_reg']
        self.k = self.config['k']
        
#         self.index = torch.arange(config['batch_size']).to(self.device)

        # define layers and loss
        self.user_embeddings_src = torch.nn.Embedding(num_embeddings=self.n_users_src,embedding_dim=self.latent_dim)
        self.item_embeddings_src = torch.nn.Embedding(num_embeddings=self.n_items_src,embedding_dim=self.latent_dim)
        self.user_embeddings_tgt = torch.nn.Embedding(num_embeddings=self.n_users_tgt,embedding_dim=self.latent_dim)
        self.item_embeddings_tgt = torch.nn.Embedding(num_embeddings=self.n_items_tgt,embedding_dim=self.latent_dim)
        self.embedding_init()

        self.bpr_loss = BPRLoss()
        self.ce_loss = criterion
        self.l2_loss = l2_loss
        self.reg_loss = EmbLoss()
        self.sp_loss = SPLoss(n_samples=self.config['batch_size'])

        # store vairables for full sort evaluation accelaeration
        self.restore_user_src = None
        self.restore_user_tgt = None
        self.restore_item_src = None
        self.restore_item_tgt = None
        self.embedding_list = None
        

        # parameter initial
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.user_centroids_src = None
        self.user_2cluster_src = None
        self.user_2cluster_dist_src = None

        self.user_centroids_tgt = None
        self.user_2cluster_tgt = None
        self.user_2cluster_dist_tgt = None

        # model definition
        self.graph = self.create_graph(self.src, self.tgt)
        self.neighbors_utable_src, self.neighbors_utable_tgt = self.neighbor_table_init()
        self.layers = nn.ModuleList()
#         for i in range(self.n_layers):
#             self.layers.append(NeighborEnhancedGCN(self.latent_dim))

        self.graph_embedding_init()
        self.net = GraphNet_CrossGCF(self.graph, self.latent_dim, [self.latent_dim] * self.n_layers, [0.1] * self.n_layers).to(self.device)

    def create_graph(self, src, tgt):
        src_start = torch.from_numpy(src['user_id'].values)
        src_end = torch.from_numpy(src['item_id'].values)
        tgt_start = torch.from_numpy(tgt['user_id'].values)
        tgt_end = torch.from_numpy(tgt['item_id'].values)

        graph_data = {
            ('u', 'src', 'i'): (src_start, src_end),
            ('i', 'src-by', 'u'): (src_end, src_start),
            ('u', 'tgt', 'i'): (tgt_start, tgt_end),
            ('i', 'tgt-by', 'u'): (tgt_end, tgt_start)
        }  

        graph = dgl.heterograph(graph_data).to(self.device)
        return graph

    def get_subgraphs(self, graph, relations):
        sub_graph = dgl.edge_type_subgraph(graph, relations)
        neighbors_table = self.neighbor_table(sub_graph, [relations[0][1], relations[1][1]]) # src, src_by
        return sub_graph, neighbors_table

    def neighbor_table(self, subgraph, relations):
        # adj = subgraph.adjacency_matrix(etype=relations[0]).to_dense() # 耗时 2s
        # adj_inverse = subgraph.adjacency_matrix(etype=relations[1]).to_dense()
        # neighbors_table = torch.matmul(adj, adj_inverse)
        adj = subgraph.adjacency_matrix(etype=relations[0]) # 耗时,使用sparse减小内存使用
        adj_inverse = subgraph.adjacency_matrix(etype=relations[1])
        neighbors_table = torch.sparse.mm(adj, adj_inverse).to_dense() # .t()
        neighbors_table = F.normalize(neighbors_table, p=2, dim=1)  # 归一化加权,改成稀疏矩阵乘法，sparse、再存一个图
        return neighbors_table

    def embedding_init(self):
        self.user_embeddings_src.weight.data = nn.Parameter(self.data_uA_graph_fea).to(self.device)
        self.user_embeddings_tgt.weight.data = nn.Parameter(self.data_uB_graph_fea).to(self.device)
        self.item_embeddings_src.weight.data = nn.Parameter(self.data_iA_graph_fea).to(self.device)
        self.item_embeddings_tgt.weight.data = nn.Parameter(self.data_iB_graph_fea).to(self.device)

    def graph_embedding_init(self):
        u_info = torch.maximum(self.user_embeddings_src.weight.data, self.user_embeddings_tgt.weight.data)
        i_info = torch.cat((self.item_embeddings_src.weight.data, self.item_embeddings_tgt.weight.data), dim=0)
        self.graph.nodes['u'].data['info'] = u_info
        self.graph.nodes['i'].data['info'] = i_info


    def neighbor_table_init(self):
        _, neighbors_utable_src = self.get_subgraphs(self.graph, [('u', 'src', 'i'), ('i', 'src-by', 'u')]) 
        _, neighbors_utable_tgt = self.get_subgraphs(self.graph, [('u', 'tgt', 'i'), ('i', 'tgt-by', 'u')])

        neighbors_utable_src,neighbors_utable_tgt = neighbors_utable_src.to(config['device']),neighbors_utable_tgt.to(config['device'])
#         _, neighbors_utable_src = dense_to_sparse(neighbors_utable_src)
#         _, neighbors_utable_tgt = dense_to_sparse(neighbors_utable_tgt)
        
        return neighbors_utable_src, neighbors_utable_tgt

    @get_time
    def cross_domain_graph_convolution(self):

        u_info = torch.maximum(self.user_embeddings_src.weight.data, self.user_embeddings_tgt.weight.data)
        i_info = torch.cat((self.item_embeddings_src.weight.data, self.item_embeddings_tgt.weight.data), dim=0)
        feat_dict = {'u': u_info, 'i': i_info}

        embedding_list = []
        for layer in self.layers: 
            src_u, tgt_u = aggravate_domain(self.neighbors_utable_src, self.neighbors_utable_tgt, u_info)
            embedding_list.append([src_u, tgt_u]) # [[],[]]

            msg_emb = layer(self.graph, feat_dict)
            u_info, i_info = torch.maximum(msg_emb['src_u'], msg_emb['tgt_u']), torch.cat((msg_emb['src_i'][0:self.index_gap, :], msg_emb['tgt_i'][self.index_gap:,:]), dim=0)

            feat_dict = {'u': u_info, 'i': i_info}
        embedding_list.append([src_u, tgt_u]) 

        src_item = i_info[:self.index_gap + 1, :]
        tgt_item = i_info[self.index_gap + 1:, :]

        return src_u, tgt_u, src_item, tgt_item, embedding_list


    def cross_domain_graph_convolution_two(self):
        embedding_list = []
        u_info, i_info, user_embeds, _ = self.net(self.graph, 'u', 'i')
        for i in range(len(user_embeds)):
            user_feature = user_embeds.pop(0)
            tmp_src_u, tmp_tgt_u = aggravate_domain_matrix(self.neighbors_utable_src, self.neighbors_utable_tgt, user_feature)
            embedding_list.append([tmp_src_u, tmp_tgt_u])
        src_u, tgt_u = aggravate_domain_matrix(self.neighbors_utable_src, self.neighbors_utable_tgt, u_info)
#         src_u, tgt_u = embedding_list[-1][0], embedding_list[-1][1]
#         src_u, tgt_u = embedding_list[-1][0], embedding_list[-1][1]
#         src_item, tgt_item = item_embeds[-1][:self.index_gap + 1, :], item_embeds[-1][self.index_gap + 1:, :]
        src_item, tgt_item = i_info[:self.index_gap + 1, :], i_info[self.index_gap + 1:, :]


        return src_u, tgt_u, src_item, tgt_item, embedding_list
    
    
    def e_step(self):
        """look for centroids"""
        user_embeddings_src = self.user_embeddings_src.weight.detach().cpu().numpy()
        user_embeddings_tgt = self.user_embeddings_tgt.weight.detach().cpu().numpy()
        indexes_src, indexes_tgt = self.overlap + self.unique_A, self.overlap + self.unique_B
        self.user_centroids_src, self.user_2cluster_src, self.user_2cluster_dist_src = self.run_kmeans(user_embeddings_src, indexes_src)
        self.user_centroids_tgt, self.user_2cluster_tgt, self.user_2cluster_dist_tgt = self.run_kmeans(user_embeddings_tgt, indexes_tgt)

    def run_kmeans(self, x, indexes):
        """get kmeans"""
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        use_emb = np.take(x, indexes, axis=0)
        kmeans.train(use_emb)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1) # D=[D,k],
        # sim = 1/(D+1e-10)

        # cos sim
        D = x.dot(cluster_cents.T) / (np.linalg.norm(x) * np.linalg.norm(cluster_cents.T))
        sim = 1-D # D*c

        # convert to cuda Tensor for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        node2cluster_dist = F.normalize(torch.from_numpy(sim).to(self.device), p=2, dim=1)

        return centroids, node2cluster, node2cluster_dist


    def proto_nce_loss(self, user_embeddings_src_all, user, batch_type, index):
        """global structual loss"""
        user_embeddings_src = user_embeddings_src_all[user] # B, e
        norm_user_embedding_src = F.normalize(user_embeddings_src) 
        if batch_type == "src":
            user2cluster_dist_tgt = self.user_2cluster_dist_tgt[user] # B, c
        else:
            user2cluster_dist_tgt = self.user_2cluster_dist_src[user] # B, c

        # user2cluster_dist_tgt = np.take_along_axis(user2cluster_dist_tgt, np.argsort(self.user_2cluster_src), axis=1) # sort
        user_embeddings_tgt_new = torch.matmul(user2cluster_dist_tgt, self.user_centroids_src) # B,e
        pos_score_user = torch.mul(norm_user_embedding_src, user_embeddings_tgt_new).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embedding_src, user_embeddings_tgt_new.transpose(0,1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
#         print(-torch.log(pos_score_user / ttl_score_user)[0])
        proto_nce_loss_user = self.proto_reg * self.sp_loss(-torch.log(pos_score_user / ttl_score_user), index) 
#         proto_nce_loss_user = self.proto_reg * torch.sum((-torch.log(pos_score_user / ttl_score_user))) 

        return proto_nce_loss_user


    def ssl_layer_loss(self, current_embedding, previous_embedding, user, index):
        """local structual loss"""
        # current_user_embeddings_all_src = current_embedding
        # previous_user_embeddings_all_src = previous_embedding
        previous_user_embeddings_src = previous_embedding[user] 
        current_user_embeddings_tgt = current_embedding[user] 

        norm_user_emb_src = F.normalize(previous_user_embeddings_src)
        norm_user_emb2_tgt = F.normalize(current_user_embeddings_tgt) # B,d
        norm_all_user_emb_tgt = F.normalize(previous_embedding) # D,d
        pos_score_user = torch.mul(norm_user_emb_src, norm_user_emb2_tgt).sum(dim=1) # B
        ttl_score_user = torch.matmul(norm_user_emb_src, norm_all_user_emb_tgt.transpose(0, 1)) # B,D
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = self.sp_loss(-torch.log(pos_score_user / ttl_score_user), index) 
#         ssl_loss_user = torch.sum(-torch.log(pos_score_user / ttl_score_user)) 

        ssl_loss = self.ssl_reg * ssl_loss_user
        return ssl_loss


    def forward(self):
        """model forward"""
        src_u, tgt_u, src_item, tgt_item, embedding_list = self.cross_domain_graph_convolution_two()
        # 多层mean
        return src_u, tgt_u, src_item, tgt_item, embedding_list

    
    def calculate_loss_bpr(self, batch_input, batch_type):
        # clear the storage variable when training
#         if self.restore_user_src is None or self.restore_item_src is None or self.restore_user_tgt is None or self.restore_item_tgt is None:
#             self.restore_user_src, self.restore_item_src, self.restore_user_tgt, self.restore_item_tgt = None, None, None, None

        user, positive_item, negative_items, ratings = batch_input
        index = torch.arange(user.shape[0]).to(self.device)
        
        if batch_type == "src":
            user_all_embeddings_src, user_all_embeddings_tgt, item_all_embeddings_src, item_all_embeddings_tgt, embeddings_list = self.forward()
        else:
            user_all_embeddings_src, item_all_embeddings_src, user_all_embeddings_tgt, item_all_embeddings_tgt, embeddings_list = self.restore_user_src, self.restore_item_src, self.restore_user_tgt, self.restore_item_tgt, self.embedding_list

        if batch_type == "src":
            # 防止一个batch二次forward
            self.restore_user_src, self.restore_item_src, self.restore_user_tgt, self.restore_item_tgt, self.embedding_list = user_all_embeddings_src, item_all_embeddings_src, user_all_embeddings_tgt, item_all_embeddings_tgt,embeddings_list
        else:
            self.restore_user_src, self.restore_item_src, self.restore_user_tgt, self.restore_item_tgt, self.embeddings_list = None, None, None, None,None


#         user_all_embeddings_src, user_all_embeddings_tgt, item_all_embeddings_src, item_all_embeddings_tgt, embeddings_list = self.forward()

        if batch_type == 'src':
            user_all_embeddings, item_all_embeddings = user_all_embeddings_src, item_all_embeddings_src
            center_embedding_all = embeddings_list[1][0]  # 1-hop
            local_embedding_all = embeddings_list[self.hyper_layers * 2][1]
        else:
            user_all_embeddings, item_all_embeddings = user_all_embeddings_tgt, item_all_embeddings_tgt
            center_embedding_all = embeddings_list[1][1]  # 1-hop
            local_embedding_all = embeddings_list[self.hyper_layers * 2][0]

        ssl_loss = self.ssl_layer_loss(local_embedding_all, center_embedding_all, user, index)  # scalar
        proto_loss = self.proto_nce_loss(center_embedding_all, user, batch_type ,index)

        u_embeddings = user_all_embeddings[user].unsqueeze(dim=1)
        pos_embeddings = item_all_embeddings[positive_item].unsqueeze(dim=1)
        neg_embeddings = item_all_embeddings[negative_items.view(-1)]
        neg_embeddings = neg_embeddings.view(-1, config['neg_num'], self.latent_dim)

        # calculate rec Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1) # batch dim

        mf_loss = self.bpr_loss(pos_scores, neg_scores)


        if batch_type == 'src':
            u_ego_embeddings = self.user_embeddings_src(user)  # reg
            pos_ego_embeddings = self.item_embeddings_src(positive_item)
            neg_ego_embeddings = self.item_embeddings_src(negative_items.view(-1))
        else:
            u_ego_embeddings = self.user_embeddings_tgt(user)  # reg
            pos_ego_embeddings = self.item_embeddings_tgt(positive_item)
            neg_ego_embeddings = self.item_embeddings_tgt(negative_items.view(-1))
            
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss , self.reg_weight * reg_loss, ssl_loss, proto_loss


    def calculate_loss(self, batch_input, batch_type):
        # clear the storage variable when training
#         if self.restore_user_src is None or self.restore_item_src is None or self.restore_user_tgt is None or self.restore_item_tgt is None:
#             self.restore_user_src, self.restore_item_src, self.restore_user_tgt, self.restore_item_tgt = None, None, None, None

        user, item, ratings = batch_input
        index = torch.arange(user.shape[0]).to(self.device)

        
#         user_all_embeddings_src, user_all_embeddings_tgt, item_all_embeddings_src, item_all_embeddings_tgt, embeddings_list = self.forward()
        if batch_type == "src":
            user_all_embeddings_src, user_all_embeddings_tgt, item_all_embeddings_src, item_all_embeddings_tgt, embeddings_list = self.forward()
        else:
            user_all_embeddings_src, item_all_embeddings_src, user_all_embeddings_tgt, item_all_embeddings_tgt, embeddings_list = self.restore_user_src, self.restore_item_src, self.restore_user_tgt, self.restore_item_tgt, self.embedding_list

        if batch_type == "src":
            # avoid second forward
            self.restore_user_src, self.restore_item_src, self.restore_user_tgt, self.restore_item_tgt, self.embedding_list = user_all_embeddings_src, item_all_embeddings_src, user_all_embeddings_tgt, item_all_embeddings_tgt,embeddings_list
        else:
            self.restore_user_src, self.restore_item_src, self.restore_user_tgt, self.restore_item_tgt, self.embeddings_list = None, None, None, None, None

        if batch_type == 'src':
            ratings = ratings / self.dataset_A.maxRate
            user_all_embeddings, item_all_embeddings = user_all_embeddings_src, item_all_embeddings_src
            center_embedding_all = embeddings_list[1][0]  
            local_embedding_all = embeddings_list[self.hyper_layers * 2][1]
        else:
            ratings = ratings / self.dataset_B.maxRate
            user_all_embeddings, item_all_embeddings = user_all_embeddings_tgt, item_all_embeddings_tgt
            center_embedding_all = embeddings_list[1][1]  
            local_embedding_all = embeddings_list[self.hyper_layers * 2][0]

        ssl_loss = self.ssl_layer_loss(local_embedding_all, center_embedding_all, user, index) # scalar
        proto_loss = self.proto_nce_loss(center_embedding_all, user, batch_type,index)

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        # norm u, i
        norm_u_embeddings, norm_i_embeddings = reg(u_embeddings), reg(i_embeddings)

        # calculate rec Loss
        scores = torch.sum(
            torch.multiply(u_embeddings, i_embeddings), axis=1,
            keepdims=False) / (norm_u_embeddings * norm_i_embeddings)
        scores = torch.clamp(scores, 1e-6)

        mf_loss = -torch.sum(self.ce_loss(scores, ratings))

        if batch_type == 'src':
            u_ego_embeddings = self.user_embeddings_src(user) # reg
            i_ego_embeddings = self.item_embeddings_src(item)
        else:
            u_ego_embeddings = self.user_embeddings_tgt(user) # reg
            i_ego_embeddings = self.item_embeddings_tgt(item)
#         reg_loss = self.reg_loss(u_ego_embeddings, i_ego_embeddings)
        reg_loss = self.l2_loss(u_ego_embeddings) + self.l2_loss(i_ego_embeddings)
        return mf_loss , self.reg_weight * reg_loss, ssl_loss, proto_loss


    def predict(self, batch_input, batch_type):
        """inference for batch"""
        user, item, _ = batch_input
        # store to avoid many forward
#         if self.restore_user_src is None or self.restore_item_src is None or self.restore_user_tgt is None or self.restore_item_tgt is None:
#             print("！")
        restore_user_src, restore_user_tgt, restore_item_src, restore_item_tgt, _ = self.forward()
        if batch_type =='src':
            user_all_embeddings = restore_user_src
            item_all_embeddings = restore_item_src
        else:
            user_all_embeddings = restore_user_tgt
            item_all_embeddings = restore_item_tgt

        u_embeddings = user_all_embeddings[user.view(-1, )] # batch_size * 100 , 32
        i_embeddings = item_all_embeddings[item.view(-1, )] # batchsize * 100, 32


        # norm u, i
#         norm_u_embeddings, norm_i_embeddings = reg(u_embeddings), reg(i_embeddings)

        # calculate rec Loss
#         scores = torch.sum(
#             torch.multiply(u_embeddings, i_embeddings), axis=1,
#             keepdims=False) / (norm_u_embeddings * norm_i_embeddings)
#         scores = torch.clamp(scores, 1e-6).view(-1, 100)
        
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=-1).squeeze().view(-1, 100) # B*100,

        return scores

    def full_sort_predict(self, batch_input, batch_type):
        """inference for all the interaction"""
        # user = interaction[self.USER_ID]
        user, _, _ = batch_input
        user = user[:,0] # batchsize, 
        restore_user_src, restore_user_tgt, restore_item_src, restore_item_tgt, _ = self.forward()
        if batch_type =='src':
            user_all_embeddings = restore_user_src
            item_all_embeddings = restore_item_src
        else:
            user_all_embeddings = restore_user_tgt
            item_all_embeddings = restore_item_tgt
    
        # get user embedding from storage
        u_embeddings = user_all_embeddings[user] # batch_size, 32
        i_embeddings = item_all_embeddings # D, 32
    
        # dot with all item embeddings
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1)) # B, D
    
        return scores





if __name__ == '__main__':
    pass


