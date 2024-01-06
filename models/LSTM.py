#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @author:TuDaCheng
# @file:LSTM_model.py
# @time:2023/03/22


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import torch

class Config(object):
    """TextRCNN模型配置参数"""
    def __init__(self, dataset, embedding):
        self.embedding_name = embedding
        self.model_name = "LSTM"
        self.train_path = dataset +"/data/train.txt"
        self.dev_path =dataset + "/data/test.txt"
        self.test_path = dataset +"/data/test.txt"
        self.aug_train_path = dataset +"/data/aug_train.txt"
        self.aug_dev_path = dataset +"/data/aug_test.txt"
        self.aug_test_path = dataset +"/data/aug_test.txt"
        self.aug_data_path =dataset + "/data/aug_all_data.txt"
        self.data_path =dataset + "/data/all_data.txt"


        self.vocab_path = dataset + '/data/'+self.embedding_name+'.pkl'    
        print(" self.vocab_path", self.vocab_path)      
        # self.aug_vocab_path = dataset + '/data/' + pretrain + '.pkl'   
        self.aug_vocab_path = dataset + '/data/'+self.embedding_name+'.pkl'                              # 词表
        self.save_path = dataset + '/save_model/' + self.model_name+"_"+self.embedding_name + '.ckpt'        # 模型训练结果       # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.dataset_pkl = dataset + '/data/dataset.pkl'   




       

        # self.aug_dataset_pkl =dataset + "/data/aug_dataset_pkl"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.class_list = [x.strip() for x in open(dataset +"/data/class.txt", encoding="utf-8").readlines()]
        

        self.padding_size = 200
        self.n_vocab = 300
        self.dropout = 0.3
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_vocab = 300  # 词表大小 在运行时赋值
        self.num_epochs = 50
        self.learning_rate = 9e-4
        self.batch_size = 128
        if embedding == 'random':
            self.embedding_pretrained = None

        elif embedding == 'embedding_SougouNews':
          # Load embeddings from a file using NumPy
            loaded_data = np.load(dataset + '/data/' + self.embedding_name+".npz", allow_pickle=True)
            # Extract the embeddings from the loaded data and convert to float32
            embeddings = loaded_data["embeddings"].astype('float32')
            # Convert NumPy array to a PyTorch tensor
            self.embedding_pretrained = torch.tensor(embeddings)
            self.n_vocab = self.embedding_pretrained.size(0)
            # Set self.embedding_pretrained to None if embedding is 'random'
        else:
            self.embedding_pretrained = None
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300  # 字向量维度
        self.hidden_size = 256  # lstm隐藏单元数
        self.num_layers = 2  # lstm层数

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.logs=""
        self.weight=0
        if config.embedding_pretrained is not None:  # 加载初始化好的预训练词/字嵌入矩阵  微调funetuning
            print(config.embedding_pretrained.size())
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            print(self.embedding)
        else:  # 否则随机初始化词/字嵌入矩阵 指定填充对应的索引
            self.embedding = nn.Embedding(config.num_vocab, config.embed, padding_idx=config.num_vocab - 1)

        # 单层双向lstm batch_size为第一维度
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers, bidirectional=True,
                            batch_first=True, dropout=config.dropout)

        self.maxpool = nn.MaxPool1d(config.padding_size)  # 沿着长度方向做全局最大池化

        # 输出层
        self.fc = nn.Linear(config.hidden_size*2 + config.embed, config.num_classes)
        
    def forward(self, x):
        content = x[0]  # [batch,seq_len]
        
        content = torch.clamp(input=content, min=0, max=2362)
        embed = self.embedding(content)  # [batch_size, seq_len, embeding]
        out, _ = self.lstm(embed)  # [batch_size,seq_len,hidden_size*2]
        out = torch.cat((embed, out), 2)  # 把词嵌入和lstm输出进行拼接 (batch,seq_len.embed+hidden_size*2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # [batch,embed+hidden_size*2,seq_len]
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out

    def get_parameters(self):
        parameters = list(self.final_layer.parameters())
        for model in self.models:
            parameters += list(model.parameters())
        return parameters
