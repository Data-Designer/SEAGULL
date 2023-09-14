#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 15:18
# @Author  : Anonymous
# @Site    : 
# @File    : ngl.py
# @Software: PyCharm

# #Desc: cross neighbor aware gcn
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from config import *
from utils import *


class CrossGCF(nn.Module):
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(CrossGCF, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        #weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias = True)
        self.W2 = nn.Linear(in_size, out_size, bias = True)
        self.W3 = nn.Linear(in_size, out_size, bias = True)
        
        self.attn_fc = nn.Linear(2 * in_size, 1, bias=False)


        #leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)
        torch.nn.init.xavier_uniform_(self.W3.weight)
        torch.nn.init.constant_(self.W3.bias, 0)
        
        torch.nn.init.xavier_uniform_(self.attn_fc.weight)


        # norm
        self.norm_dict = norm_dict
        
    def edge_attention(self, edges):
        emb = torch.cat([edges.src['info'], edges.dst['info']], dim=1)
        a = self.attn_fc(emb)
        return {'a': self.leaky_relu(a)}

    def forward(self, g, feat_dict):
        funcs = {} # message and reduce functions dict
        
        # for each type of edges, compute messages and reduce them all
        for srctype, etype, dsttype in g.canonical_etypes: # 0.0017
            if srctype == dsttype: # for self loops
                messages = self.W1(feat_dict[srctype])
                g.nodes[srctype].data[etype] = messages   # store in ndata
                funcs[(srctype, etype, dsttype)] = (fn.copy_u(etype, 'm'), fn.sum('m', 'h'))  #define message and reduce functions
            else:
                src, dst = g.edges(etype=(srctype, etype, dsttype)) 

                src, dst = src.long(), dst.long() 

                norm = self.norm_dict[(srctype, etype, dsttype)] 
                messages = norm * (self.W1(feat_dict[srctype][src]) + self.W2(
                    feat_dict[srctype][src] * feat_dict[dsttype][dst]))  # compute messages， B*D 
                                
                g.edges[(srctype, etype, dsttype)].data[etype] = messages  


                g.apply_edges(self.edge_attention,etype=(srctype, etype, dsttype))
                e = g.edges[(srctype, etype, dsttype)].data['a']
                g.edges[(srctype, etype, dsttype)].data['a'] = edge_softmax(g[(srctype, etype, dsttype)], e) 
                # a*e
                g.edges[(srctype, etype, dsttype)].data[etype] = torch.mul(g.edges[(srctype, etype, dsttype)].data.pop('a'), g.edges[(srctype, etype, dsttype)].data.pop(etype))

                

                
                funcs[(srctype, etype, dsttype)] = (fn.copy_e(etype, 'm'), fn.sum('m', 'h'))  

        g.multi_update_all(funcs, 'sum') 

        feature_dict={}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data['h'])
            h = self.dropout(h) 
            h = F.normalize(h,dim=1,p=2) 
            feature_dict[ntype] = h
        return feature_dict



class GraphNet_CrossGCF(nn.Module):
    def __init__(self, g, in_size, layer_size, dropout, lmbd=1e-5):
        super(GraphNet_CrossGCF, self).__init__()
        self.lmbd = lmbd
        self.norm_dict = dict()
        for srctype, etype, dsttype in g.canonical_etypes:
            src, dst = g.edges(etype=(srctype, etype, dsttype))
            dst_degree = g.in_degrees(dst, etype=(srctype, etype, dsttype)).float() 
            src_degree = g.out_degrees(src, etype=(srctype, etype, dsttype)).float()
            norm = torch.pow(src_degree * dst_degree, -0.5).unsqueeze(1) 
            self.norm_dict[(srctype, etype, dsttype)] = norm

        self.layers = nn.ModuleList()
        self.layers.append(
            NGCFLayer(in_size, layer_size[0], self.norm_dict, dropout[0])
        )
        self.num_layers = len(layer_size)
        for i in range(self.num_layers-1):
            self.layers.append(
                NGCFLayer(layer_size[i], layer_size[i+1], self.norm_dict, dropout[i+1])
            )
        self.initializer = nn.init.xavier_uniform_

        #  embeddings for different types of nodes
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(g.nodes[ntype].data['info']) for ntype in g.ntypes
        })

    def forward(self, g, user_key, item_key):
        h_dict = {ntype : self.feature_dict[ntype] for ntype in g.ntypes} # u: D*e, i:D*e
        # obtain features of each layer and concatenate them all
        user_embeds = []
        item_embeds = []
        user_embeds.append(h_dict[user_key])
        item_embeds.append(h_dict[item_key])
        for layer in self.layers:
            h_dict = layer(g, h_dict)
            user_embeds.append(h_dict[user_key])
            item_embeds.append(h_dict[item_key])
        user_embd = torch.mean(torch.stack(user_embeds), 0) # 3*H
        item_embd = torch.mean(torch.stack(item_embeds), 0)
        return user_embd, item_embd, user_embeds, item_embeds







