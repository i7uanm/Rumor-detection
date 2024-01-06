# -*- coding: utf-8 -*-
# @Time : 2023/3/4 17:17
# @Author : TuDaCheng
# @File : data_utils.py

import re
import os
import torch
import pickle
import time
from tqdm import tqdm
from datetime import timedelta

UNK, PAD = "UNK", "PAD"


def preprocessing_text(text):
    """
    文本预处理 去出停用词和非中文其他符号
    :param text:
    :return:
    """

    def remove_1a():
        # 去除标点字母数字
        chinese = '[\u4e00-\u9fa5]+'
        str1 = re.findall(chinese, text)
        text2 = ''.join(str1)
        return text2

    def delete_boring_characters():
        sentence = remove_1a()
        return re.sub(r'[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", sentence)

    def get_stopword_list():
        stop_word_path = './data/stopword.txt'
        stop_word_list_ = [sw.replace('\n', '') for sw in open(stop_word_path, encoding='utf-8').readlines()]
        return stop_word_list_

    content = delete_boring_characters()
    stop_word_list = get_stopword_list()
    return content, stop_word_list


def build_vocab(config):
    """
    构建词字典
    :param file_path:
    :param max_size:
    :param min_freq:
    :return:
    """
    file_path = config.data_path
    vocab_dict = {}
    max_size = 10000
    min_freq = 1
    # 对句子进行分字处理
    print(config.embedding_name)
    if config.embedding_name == "roberta":
        from transformers import BertTokenizer
        # tokenizer = RobertaTokenizer.from_pretrained('roberta-chinese')
        # tokenizer = BertTokenizer.from_pretrained('thu-coai/roberta-zh-specific')
        from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
        # tokenizer = RobertaTokenizer.from_pretrained("./models/Roberta")
        try:
    # 尝试从本地加载模型和tokenizer
            ro_config = RobertaConfig.from_pretrained("./models/Roberta")
            tokenizer = RobertaTokenizer.from_pretrained("./models/Roberta", config=ro_config)
        except:
            # 如果本地不存在，从在线加载
            ro_config = RobertaConfig.from_pretrained("roberta-base")
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base", config=ro_config)
        with open(file_path, "r", encoding="utf-8") as f_reader:
            for line in tqdm(f_reader):
                lens = len(line.strip().split("\t"))
                if lens == 2:
                    label, sentence = line.strip().split("\t")
                    sentence, stop_word_l = preprocessing_text(sentence)
                    for token in tokenizer.tokenize(sentence):
                        vocab_dict[token] = vocab_dict.get(token, 0) + 1  # 统计字频
                # 过来低频词排序 取出max_size个单词
            vocab_list = sorted([item for item in vocab_dict.items() if item[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[:max_size]
                # 构建字典映射
            vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
            vocab_dict.update({UNK: len(vocab_dict), PAD: len(vocab_dict)+1})
    elif config.embedding_name == 'random':
            tokenizer = lambda s: [w for w in s]
            with open(file_path, "r", encoding="utf-8") as f_reader:
                for line in tqdm(f_reader):
                    lens = len(line.strip().split("\t"))
                    if lens == 2:
                        label, sentence = line.strip().split("\t")
                        sentence, stop_word_l = preprocessing_text(sentence)
                        for word in tokenizer(sentence):
                            vocab_dict[word] = vocab_dict.get(word, 0) + 1  # 统计字频
                # 过来低频词排序 取出max_size个单词
                vocab_list = sorted([item for item in vocab_dict.items() if item[1] >= min_freq],
                                key=lambda x: x[1], reverse=True)[:max_size]
                # 构建字典映射
                vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
                vocab_dict.update({UNK: len(vocab_dict), PAD: len(vocab_dict)+1})
    # elif config.embedding ==
    elif config.embedding_name == "M3e":
        from transformers import AutoTokenizer
        tokenizer_path = "./data/tokenizer.json"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        with open(file_path, "r", encoding="utf-8") as f_reader:
            for line in tqdm(f_reader):
                lens = len(line.strip().split("\t"))
                if lens == 2:
                    label, sentence = line.strip().split("\t")
                    sentence, stop_word_l = preprocessing_text(sentence)
                    for token in tokenizer.tokenize(sentence):
                        vocab_dict[token] = vocab_dict.get(token, 0) + 1  # 统计字频
                # 过来低频词排序 取出max_size个单词
            vocab_list = sorted([item for item in vocab_dict.items() if item[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[:max_size]
                # 构建字典映射
            vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
            vocab_dict.update({UNK: len(vocab_dict), PAD: len(vocab_dict)+1})
    with open("./data/"+config.embedding_name+".pkl", "wb") as f_writer:
        pickle.dump(vocab_dict, f_writer)
    print("save as ./data/{}.pkl".format(config.embedding_name))
    n_vocab = len(vocab_dict)
    return vocab_dict,n_vocab



def get_dict(path):
    """
    加载字典
    :param path:
    :return:
    """
    with open(path, "rb") as f_reader:
        vocab_dict = pickle.load(f_reader)
    return vocab_dict


def load_dataset(file_path, config):
    print(load_dataset)
    """
    :param file_path:
    :param config:
    :return:
    """
    # 对句子进行分字处理
    tokenizer = lambda s: [w for w in s]
    contents = []
    vocab = get_dict(config.vocab_path)
    with open(file_path, "r", encoding="utf-8") as f_reader:
        for line in f_reader:
            lengths = len(line.strip().split("\t"))
            if lengths == 2:
                    label, text = line.strip().split("\t")
                    words_line = []  # word to id
                    text, _ = preprocessing_text(text)
                    token = tokenizer(text)
                    seq_length = len(token)
                    if len(token) < config.padding_size:
                        token.extend(["PAD"] * (config.padding_size - len(token)))
                    else:
                        token = token[: config.padding_size]
                        seq_length = config.padding_size
                    # word2id
                    for word in token:
                        words_line.append(vocab.get(word, vocab.get(UNK)))
                    print('aaa',words_line)
                    contents.append((words_line, int(label), seq_length))

    return contents


def build_dataset(config,is_Enhance=False):
    """
    构建数据集
    :param config:
    :return:
    """

    if is_Enhance:
        print('is enhance')
        # 构建字典
        if os.path.exists(config.aug_vocab_path):
            aug_vocab_dict = pickle.load(open(config.aug_vocab_path, "rb"))
        else:
            aug_vocab_dict,aug_n_vocab  = build_vocab(config.aug_data_path)
        print("aug_vocab length", len(aug_vocab_dict))
        # 分别对 训练集 验证集 测试集进行处理 把文本中的词转化成字典中的索引id
        aug_train_data = load_dataset(config.aug_train_path, config)  # (words_line, int(label), seq_length)
        aug_dev_data = load_dataset(config.aug_dev_path, config)
        aug_test_data = load_dataset(config.aug_test_path, config)
        dataset = {}
        dataset["aug_train_data"] = aug_train_data
        dataset["aug_dev_data"] = aug_dev_data
        dataset["aug_test_data"] = aug_test_data
        pickle.dump(dataset, open(config.aug_dataset_pkl, "wb"))
        return aug_vocab_dict, aug_train_data, aug_dev_data, aug_test_data
    else:
        print('is not enhance')
        print('config.vocab_path',config.vocab_path)
        if os.path.exists(config.vocab_path):
            print('is exist')
            vocab_dict = pickle.load(open(config.vocab_path, "rb"))
            n_vocab = len(vocab_dict)
            print("n_vocab",n_vocab)
            config.n_vocab = n_vocab
        else:
            print('is not exist')
            vocab_dict,n_vocab  = build_vocab(config)
            
        print("vocab length", len(vocab_dict))
        config.n_vocab = len(vocab_dict)
        print(config.n_vocab)
        if os.path.exists(config.dataset_pkl):
            print(2)
            dataset = pickle.load(open(config.dataset_pkl, "rb"))
            train_data = dataset["train_data"]
            dev_data = dataset["dev_data"]
            test_data = dataset["test_data"]
            pass
        else:
            print('create dataset')
            # 分别对 训练集 验证集 测试集进行处理 把文本中的词转化成字典中的索引id
            train_data = load_dataset(config.train_path, config)  # (words_line, int(label), seq_length)
            # with open(config.train_path, 'r', encoding='utf-8') as file:
            #     for line in file:
            #         print(line.strip())

            dev_data = load_dataset(config.dev_path, config)
            test_data = load_dataset(config.test_path, config)
            dataset = {}
            dataset["train_data"] = train_data
            dataset["dev_data"] = dev_data
            dataset["test_data"] = test_data
            pickle.dump(dataset, open(config.dataset_pkl, "wb"))

    return vocab_dict, train_data, dev_data, test_data


class DatasetIterater(object):  # 自定义数据集迭代器
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches  # 构建好的数据集
        self.n_batches = len(batches) // batch_size  # 得到batch数量
        print("len(batches)",len(batches))
        print("batch_size",batch_size)
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:  # 不能整除
            self.residue = True  # True表示不能整除
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # 转换为tensor 并 to(device)
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # seq_len为文本的实际长度（不包含填充的长度） 转换为tensor 并 to(device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:  # 当数据集大小不整除 batch_size时，构建最后一个batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)  # 把最后一个batch转换为tensor 并 to(device)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:  # 构建每一个batch
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)  # 把当前batch转换为tensor 并 to(device)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:  # 不能整除
            return self.n_batches + 1  # batch数+1
        else:
            return self.n_batches


# def build_iterator(dataset, config):  # 构建数据集迭代器
#     iter = DatasetIterater(dataset, config.batch_size, config.device)
#     return iter

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter



def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, *value, end="\n"):

        current = time.strftime("[%Y-%m-%d %H:%M:%S]")
        s = current
        for v in value:
            s += " " + str(v)
        s += end
        print(s, end="")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(s)


if __name__ == '__main__':
    path = "./data/all_data.txt"
    vocab_dic,n_vocab  = build_vocab(path)
    with open("./data/roberta.pkl", "rb") as f_reader:
        data = pickle.load(f_reader)
    print(len(data))

    word = "你"
    print(data.get(word, data.get(UNK)))
    print(data)

    text = "11111aaaaaa你好啊，?★、…【】《》？"
    con, stop_word_list = preprocessing_text(text)
    #
    # config = config
    # build_dataset(config)