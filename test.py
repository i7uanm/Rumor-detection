import os
import torch
import time
import torch.nn as nn
from utils.data_utils import Logger
from utils.metric import Metric
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from importlib import import_module
# from models.ensembleTrainEnhanced import EnsembleTrainEnhanced
criterion = CrossEntropyLoss()  # 多分类

import numpy as np
from importlib import import_module
from importlib import import_module
from utils.data_utils import build_dataset, build_iterator, get_time_dif
model_name = "enhancedEnsembleModel[0.7,0.1,0.1, 0.1]"
models = []
configs = []
name = ""
embedding_path = "datas/data/embedding_SougouNews.npz"


name = name[:-1]  # 去掉末尾的下划线
name = name.replace(":", "-")  # 避免文件名中出现冒号
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

            # 加载数据
aug_vocab, aug_train_data, aug_dev_data, aug_test_data = build_dataset(configs[0])
x = import_module("models." + model_name)
config = x.Config("datas/data/embedding_SougouNews.npz")
test_iter = build_iterator(aug_test_data, config)

num_labels = 2
metric = Metric(num_labels)
model = EnsembleTrainEnhanced(configs, models, num_models=len(configs), pretrain_embedding=pretrain_embedding)
    # test
model.load_state_dict(torch.load(r"D:\study\NN\LSTM-Rumor-detection\Rumor-detection6\Rumor-detection\datas\save_model\en.ckpt"))
model.eval()
logger_test = Logger(os.path.join("", "log-test-en.txt"))
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

need_labels = "datas\data\class.txt"
    # print each class
for label, precision, recall, f1 in zip(need_labels, precisions, recalls, f1s):
    s = "Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Class: {:s}".format(precision, recall, f1, label)
    logger_test.log(s)

logger_test.log()
s = "Micro Average - Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(micro_precision, micro_recall, micro_f1)
logger_test.log(s)

s = "Macro Average - Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(macro_precision, macro_recall, macro_f1)
logger_test.log(s)