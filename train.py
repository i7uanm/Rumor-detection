# -*- coding: utf-8 -*-
# @Time : 2023/3/2 15:59
# @Author : TuDaCheng
# @File : train.py
import os
import torch
import time
import torch.nn as nn
from utils.data_utils import Logger
from utils.metric import Metric
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from importlib import import_module
#logger = Logger(os.path.join("./datas/log", "log-train-LSTM.txt"))
from utils.data_utils  import build_dataset
from datetime import datetime
criterion = CrossEntropyLoss()  # 多分类

num_labels = 2
metric = Metric(num_labels)


# 基于方差缩放的参数初始化
# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                elif method == "orthogonal":  ## 正交初始化
                    nn.init.orthogonal(w, gain=1)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass



def train(config, model, train_iter, dev_iter, test_iter,return_output=False):
    current_time = datetime.now().strftime("%m.%d %H%M")
    logger = Logger(os.path.join(config.log_path, "log-train-{}-{}-{}.txt".format(config.model_name,config.embedding_name,current_time)))
    logger.log()
    logger.log("model_name:", config.model_name)
    logger.log("pretrain model name:", config.embedding_name)
    logger.log("Device:", config.device)
    logger.log("Epochs:", config.num_epochs)
    logger.log("Batch Size:", config.batch_size)
    logger.log("Learning Rate:", config.learning_rate)
    logger.log("dropout", config.dropout)
    logger.log("Max Sequence Length:", config.padding_size)
    logger.log()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0  # 记录进行到多少batch
    flag = False  # 记录是否很久没有效果提升

    for epoch in range(config.num_epochs):
        step = 0
        total=0
        accuracy = 0
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            step += 1
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            predicted = torch.max(outputs.data,1)[1]
            accuracy += (predicted == labels).sum()
            acc = float(accuracy ) / float(total)# 除以元素总数，可以用其他方式获取
            s = "Train Epoch: {:d} Step: {:d} Loss: {:.6f} accuracy:{:.3f}".format(epoch, step, loss.item(),acc)
            logger.log(s)
            if total_batch % 50 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                precision, recall, f1, improved, stop = evaluate(model, dev_iter)
                s = "Eval Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(precision, recall, f1)
                logger.log(s)

                if improved:
                    # torch.save(model.state_dict(),"./save_model/{}-.ckpt".format(config.model_name))
                    torch.save(model.state_dict(),config.save_path)
                    logger.log("Improved! Best F1: {:.4f}".format(f1))
                    logger.log()
                if stop:
                    logger.log("STOP! NO PATIENT!")
                    flag = True
                    break

                model.train()
            total_batch += 1
        if flag:
            break
    test(config, model, test_iter)
    if (return_output):
        return acc,loss

def evaluate(model, data_iter):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            labels = labels.float()
            loss = criterion(outputs, labels.long())
            loss_total += loss

            truth = map(int, labels.tolist())
            pred = outputs.softmax(dim=-1).cpu().tolist()  # 多分类
            metric.store(pred, truth)

    # 获得微平均指标
    _, _, (precision, recall, f1), _ = metric.precision_recall_f1()

    improved, stop = metric.is_improved(f1)

    return precision, recall, f1, improved, stop


def test(config, model, test_iter):
    #logger_test = Logger(os.path.join(config.log_path, "log-test-LSTM.txt"))
    
    current_time = datetime.now().strftime("%m.%d %H%M")
    logger_test = Logger(os.path.join(config.log_path, "log-test-{}-{}-{}.txt".format(config.model_name,config.embedding_name,current_time)))
    logger_test.log("model_name:", config.model_name)
    logger_test.log("pretrain model name:", config.embedding_name)
    logger_test.log("Device:", config.device)
    logger_test.log("Epochs:", config.num_epochs)
    logger_test.log("Batch Size:", config.batch_size)
    logger_test.log("Learning Rate:", config.learning_rate)
    logger_test.log("dropout", config.dropout)
    logger_test.log("Max Sequence Length:", config.padding_size)
    logger_test.log()

    # test
    # model.load_state_dict(torch.load("./save_model/{}-Aug.ckpt".format(config.model_name)))
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    loss_total = 0
    with torch.no_grad():
        for texts, labels in test_iter:
            outputs = model(texts)
            labels = labels.float()
            loss = criterion(outputs, labels.long())
            loss_total += loss

            truth = map(int, labels.tolist())
            pred = outputs.softmax(dim=-1).cpu().tolist()  # 多分类
            metric.store(pred, truth)

        (precisions, recalls, f1s), (macro_precision, macro_recall, macro_f1), \
        (micro_precision, micro_recall, micro_f1), _ = metric.precision_recall_f1()

    need_labels = config.class_list
    # print each class
    for label, precision, recall, f1 in zip(need_labels, precisions, recalls, f1s):
        s = "Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Class: {:s}".format(precision, recall, f1, label)
        logger_test.log(s)

    logger_test.log()
    s = "Micro Average - Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(micro_precision, micro_recall, micro_f1)
    logger_test.log(s)

    s = "Macro Average - Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(macro_precision, macro_recall, macro_f1)
    logger_test.log(s)
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

def ensemble_test(configs, models, name, test_iter):
    """测试集成模型，并将集成模型的训练过程和加权参数log存储。"""
    device = configs[0].device

    # 记录模型名称及权重
    log_file = open(f"{name}_logs_test.txt", "w")
    for i, model in enumerate(models):
        log_file.write(f"Model {i}: {type(model).__name__}\n")
        log_file.write(f"Weight: {model.weight:.4f}\n\n")
    log_file.close()

    # 加载数据集
    vocab, _, _, _ = build_dataset(configs[0])
    embedding_path = "datas/data/embedding_SougouNews.npz"
    # 加载预训练词向量，如果有
    if len(embedding_path) > 0:
        pretrain_embedding = np.load(embedding_path)
    else:
        pretrain_embedding = None

    # 加载加权平均模型参数
    state_dicts = [model.state_dict() for model in models[:-1]]
    ensemble_model = EnsembleModel(configs, len(models) - 1, state_dicts, pretrain_embedding).to(device)

    # 打印加权平均模型的权重
    print("Weight: {:.4f}".format(ensemble_model.weight.item()))

    # 计算测试集上的预测结果和真实标签，并计算准确率
    total_loss = 0.0  # 记录所有批次的累计损失
    total_acc = 0.0  # 记录所有批次的累计准确率
    num_batches = 0  # 记录批次的数量

    with torch.no_grad():
        for x, y in test_iter:
            x = x
            y = y

            y_pred = ensemble_model(x)  # 前向传播得到预测结果
            loss = nn.CrossEntropyLoss()(y_pred, y)  # 计算批次损失
            total_loss += loss.item() * len(x)  # 将批次损失累加到总损失中
            total_acc += (y_pred.argmax(dim=1) == y).float().sum().item()  # 将批次准确率累加到总准确率中
            num_batches += 1  # 增加批次数量计数器

    test_loss = total_loss / len(test_iter)  # 计算测试集平均损失
    test_acc = total_acc / len(test_iter)  # 计算测试集平均准确率
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 将集成模型训练过程和加权参数log存储到文件中
    log_file = open(f"{name}_logs.txt", "a")
    # 将模型权重记录到日志文件中
    weights = [param.item() for param in ensemble_model.parameters()]
    log_file.write(f"Weights: {weights}\n")
    log_file.write(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n\n")
    for i, config in enumerate(configs):
        log_file.write(f"Model {i}: {type(models[i]).__name__}({config.__str__()})\n\n")
        log_file.write(f"{models[i].logs}\n\n")
    log_file.close()