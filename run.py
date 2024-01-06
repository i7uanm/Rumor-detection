import torch
import time
import argparse
import numpy as np
from importlib import import_module
from utils.data_utils import build_dataset, build_iterator, get_time_dif
from train import train
# from models.EnsembleTrain import EnsembleTrain
#from test import test  # 加入 test 模块
from sklearn.ensemble import GradientBoostingRegressor
# from models.ensembleTrainEnhanced import EnsembleTrainEnhanced
import warnings
warnings.filterwarnings("ignore")  # 忽略UserWarning兼容性警告

# 参数配置
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# enhancedEnsembleModel(TextCNN,Transformer,LSTM,TextRNN_Att)
parser = argparse.ArgumentParser(description="Chinese Text Classification")
# parser.add_argument("--model", type=str, default="enhancedEnsembleModel(TextCNN,Transformer,LSTM,TextRNN_Att)", help="choose a model: TextCNN, LSTM, Transformer, TextRCNN, TextRNN, TextRNN_Att, Bert,M3e  ")
parser.add_argument("--model", type=str, default="TextCNN", help="choose a model: TextCNN, LSTM, Transformer, TextRCNN, TextRNN, TextRNN_Att, Bert,M3e")
parser.add_argument("--embedding", default="bert", type=str, help="random,sougou,roberta,bert")
# parser.add_argument("--pre", default="roberta", type=str, help="none, roberta, m3e, bert")
parser.add_argument("--word", default=False, type=bool, help="True for word, False for char")
args = parser.parse_args()
import torch.nn as nn
from importlib import import_module
import torch
import torch.nn as nn
if __name__ == '__main__':
    print(torch.cuda.is_available())
    model_name = args.model
    embedding = args.embedding
    if embedding == 'sougou':
        embedding = 'embedding_SougouNews'
        embedding_path = "embedding_SougouNews.npz"
    elif embedding == 'roberta':
        embedding_path = "roberta.pkl"
    elif embedding == 'random':
            embedding_path = 'random'
    elif embedding == 'bert':
         embedding_path = "bert.pkl"
    elif embedding == 'M3e':
         embedding_path = "M3e.pkl"
    if model_name[:13] == "EnsembleModel":

        models = []
        configs = []
        name = ""
        # # embedding_path = "datas/data/embedding_SougouNews.npz"
        # embedding_path = "./data/roberta.npz"
        model_strs = model_name[model_name.index("(") + 1: model_name.index(")")].split(",")

        for i, model_str in enumerate(model_strs):
            name += model_str + "_"
            model_module = import_module(f"models.{model_str}")
            config = model_module.Config(embedding_path)
            configs.append(config)
            model = model_module.Model(config).to(config.device)
            models.append(model)

        name = name[:-1]  # 去掉末尾的下划线
        name = name.replace(":", "-")  # 避免文件名中出现冒号
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 加载数据
        vocab, train_data, dev_data, test_data = build_dataset(configs[0])
        print("vocab:{}".format(len(vocab)))
        print("train_data:{}".format(len(train_data)))
        print("dev_data:{}".format(len(dev_data)))
        print(vocab)
        print(train_data)
        print(dev_data)
        if len(embedding_path) > 0:
            pretrain_embedding = np.load(embedding_path)
        else:
            pretrain_embedding = None

        # 初始化训练对象并训练模型
        trainer = EnsembleTrain(configs, models, num_models=len(configs), pretrain_embedding=pretrain_embedding)
        best_acc, best_recall, best_F1, test_pre,best_weights = trainer.train(train_data, dev_data, batch_size=128, lr=5.2e-4,
                                                                     weight_decay=1e-7,
                                                                     num_epochs=100)
        print(f"Best accuracy on dev set: {best_acc:.4f}")
        print(f"Best Recall on dev set: {best_recall:.4f}")
        print(f"Best F1 on dev set: {best_F1:.4f}")
        print(f"Best Precision on dev set: {test_pre:.4f}")
        print(f"Best weights: {best_weights}")
    elif model_name[:21] == "enhancedEnsembleModel":
            models = []
            configs = []
            name = ""
            # # embedding_path = "datas/data/embedding_SougouNews.npz"
            # embedding_path = "datas/data/roberta.pkl"
            model_strs = model_name[model_name.index("(") + 1: model_name.index(")")].split(",")

            for i, model_str in enumerate(model_strs):
                name += model_str + "_"
                model_module = import_module(f"models.{model_str}")
                config = model_module.Config('.',embedding)
                configs.append(config)
                model = model_module.Model(config).to(config.device)
                models.append(model)

            name = name[:-1]  # 去掉末尾的下划线
            name = name.replace(":", "-")  # 避免文件名中出现冒号
            np.random.seed(1)
            torch.manual_seed(1)
            torch.cuda.manual_seed(1)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # 加载数据
            aug_vocab, aug_train_data, aug_dev_data, aug_test_data = build_dataset(configs[0],True)
            print("aug-vocab:{}".format(len(aug_vocab)))
            print("aug_train_data:{}".format(len(aug_train_data)))
            print("aug_dev_data:{}".format(len(aug_dev_data)))

            if len(embedding_path) > 0:
                pretrain_embedding = np.load(embedding_path)
            else:
                pretrain_embedding = None

            # 初始化训练对象并训练模型

            trainer = EnsembleTrainEnhanced(configs, models, num_models=len(configs), pretrain_embedding=pretrain_embedding)
            best_acc, best_recall,best_F1,test_pre,best_weights = trainer.train(
                aug_train_data, aug_dev_data, lr=1e-3, batch_size=128, weight_decay=1e-7,
                                                   num_epochs=9)
            print(f"Best accuracy on dev set: {best_acc:.4f}")
            print(f"Best Recall on test set: {best_recall:.4f}")
            print(f"Best F1 on test set: {best_F1:.4f}")
            print(f"Best Precision on test set: {test_pre:.4f}")
            print(f"Best weights: {best_weights}")
    else:
        x = import_module("models." + model_name)
        print("training single model")
        # embedding = ""
        # embedding = "roberta.pkl"
        # embedding = "embedding_SougouNews.npz"

        # if args.embedding == 'random':
        #     embedding = 'random'
        config = x.Config('.',embedding)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        start_time = time.time()
        vocab, train_data, dev_data, test_data = build_dataset(config)
        print("vocab:{}".format(len(vocab)))
        print("train_data:{}".format(len(train_data)))
        print("dev_data:{}".format(len(dev_data)))

        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)


        time_dif = get_time_dif(start_time)
        print("模型开始之前 准备时间：", time_dif)

        config.num_vocab = len(vocab)

        # if model_name == 'Transformer_model':
        #     model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=config.num_classes)
        # else:
        model = x.Model(config).to(config.device)

        # for batch_idx, batch in enumerate(train_iter):
        #     print(batch[0][0].shape)
        #     print(batch[0][1].shape)
        #     trains, labels = batch.text, batch.label
        #     if trains.shape[1] != 100:
        #         trains = trains[:, :100]  # 将输入张量的长度限制在100以内
        #         labels = labels[:100]  # 将标签张量的长度限制在100以内
        train(config, model, train_iter, dev_iter, test_iter)
