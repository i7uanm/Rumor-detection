# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.data_path = dataset + '/data/all_data.txt'         
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/test.txt'                                    # 验证集
        self.embedding = embedding
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/'+embedding+'.pkl'    
        print(" self.vocab_path", self.vocab_path)      
        # self.aug_vocab_path = dataset + '/data/' + pretrain + '.pkl'   
        self.aug_vocab_path = dataset + '/data/'+embedding+'.pkl'                              # 词表
        self.save_path = dataset + '/save_model/' + self.model_name+"_"+embedding + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.dataset_pkl = dataset + '/data/dataset.pkl'   
        # Check if the embedding is not set to 'random'
        if embedding == 'embedding_SougouNews':
            print("is sougou")
          # Load embeddings from a file using NumPy
            loaded_data = np.load(dataset + '/data/' + embedding+".npz", allow_pickle=True)
            # Extract the embeddings from the loaded data and convert to float32
            embeddings = loaded_data["embeddings"].astype('float32')
            # Convert NumPy array to a PyTorch tensor
            self.embedding_pretrained = torch.tensor(embeddings)
            self.n_vocab = self.embedding_pretrained.size(0)
            print("vocab22:",self.n_vocab)
            # Set self.embedding_pretrained to None if embedding is 'random'
        else:
            self.embedding_pretrained = None

            

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.padding_size = 200
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 100         # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        print("the model using {} for pretraining".format(config.embedding))
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
