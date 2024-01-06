from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Config(object):
    """TextRCNN模型配置参数"""
    def __init__(self, dataset, embedding):
        self.embedding_name = embedding
        self.model_name = "Roberta"
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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.padding_size = 200
        self.dropout = 0.1                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs =100                                            # epoch数
        self.batch_size = 64                                        # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-4                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 100         # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256     



from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()

            # 使用 RoBERTa 模型的配置
        try:
    # 尝试从本地加载模型和tokenizer
            roberta_config = RobertaConfig.from_pretrained("./models/Roberta")
        except:
            # 如果本地不存在，从在线加载
            roberta_config = RobertaConfig.from_pretrained("roberta-base")
        # self.tokenizer = RobertaTokenizer.from_pretrained("./models/Roberta")
            # 创建 RoBERTa 模型
        self.roberta = RobertaModel(roberta_config)
        self.fc = nn.Linear(roberta_config.hidden_size, 2)
        # self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        print("Simple RoBERTa model created.")
    # def forward(self, input_text):
    #     print(type(input_text))
    #     print(type(input_text[0]))
    #     # 获取 RoBERTa 模型的输出
    #     inputs = self.tokenizer(input_text[0], return_tensors="pt", truncation=True, padding=True)

    #     # 获取 RoBERTa 模型的输出
    #     outputs = self.roberta(**inputs)
    #     # outputs = self.roberta(input_ids[0], attention_mask=attention_mask)

    #     outputs = outputs.last_hidden_state[:, 0, :]  # 取CLS token的表示

    #     return outputs

    def forward(self, input_ids, attention_mask=None):
        # 获取 RoBERTa 模型的输出
        outputs = self.roberta(input_ids[0], attention_mask=attention_mask)

        outputs = outputs.last_hidden_state[:, 0, :]  # 取CLS token的表示

        return outputs
