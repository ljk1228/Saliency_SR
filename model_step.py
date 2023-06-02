"""
step多标签分类模型
"""

import torch
import torch.nn as nn
from torch.nn import MultiLabelSoftMarginLoss
from torch.nn import CrossEntropyLoss
#from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.common.util import pad_sequence_to_length
from transformers import BertModel
from modeling_bert1 import BertModel1
import config_step as config

import pickle
import numpy

def get_type_rep():

    bert = BertModel.from_pretrained(config.bert_dir)

    text = [['[CLS]'] + x.split('_')[1:] for x in config.step_type]
    sub_idx = [config.tokenizer.convert_tokens_to_ids(x) for x in text]
    sub_idx = list(map(lambda x: pad_sequence_to_length(x, 5), sub_idx))
    sub_idx = torch.LongTensor(sub_idx)

    outputs= bert(sub_idx)
    rep=outputs['pooler_output'] #pooler_output：shape是(batch_size, hidden_size)，这是序列的第一个token(classification token)的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的。（通常用于句子分类，至于是使用这个表示，还是使用整个输入序列的隐藏状态序列的平均化或池化，视情况而定）
    #print(rep.shape) #bert_base 隐藏层层数768，bert_large 隐藏层层数1024
    return rep

#多标签模型
class SentenceClassification(nn.Module):
    def __init__(self, bert_dir, y_num,type_rep):#构造函数
        super().__init__()

        self.y_num = y_num
        self.bert = BertModel.from_pretrained(bert_dir) #模型
        self.type_rep = type_rep
        self.type_rep.requires_grad = False
        self.fc = nn.Linear(self.bert.config.hidden_size+2, self.y_num)

    def getName(self):
        return self.__class__.__name__

    #logits就是最终的全连接层的输出，神经网络中都是先有logits，而后通过sigmoid函数或者softmax函数得到概率的
    def compute_logits(self, data_x, bert_mask, offset1, pro1):
        outputs = self.bert(data_x, attention_mask=bert_mask)
        bert_enc = outputs[1] #bert输出的文本表示 这是序列的第一个token(classification token)的最后一层的隐藏状态 hiden_size=768

        ofs1 = offset1.unsqueeze(1)  #位置特征表示
        pro1 = pro1.unsqueeze(1)
        bert_enc = torch.cat([bert_enc, ofs1, pro1], dim=1) #文本表示和位置特征表示按行拼接在一起作为句子表示 hiden_size=770
        logits = self.fc(bert_enc)
        """

        #事件类型T的one-hot表示来代替句子级事件标签gs来进行评估，表示相对于事件类型T的单词级显著性
        
        dim1, dim3 = bert_enc.size()
        dim2 = self.y_num
        bert_enc = bert_enc.unsqueeze(1).expand_as(torch.randn(dim1, dim2, dim3))            #[1,8,768]
        type_rep_x = self.type_rep.unsqueeze(0).expand_as(torch.randn(dim1, dim2, dim3))
        bert_enc = bert_enc + type_rep_x

        # 先代替完在加位置信息
        ofs1 = offset1.unsqueeze(1).unsqueeze(2).expand_as(torch.randn(dim1, dim2, 1)) #[1,8,1]
        pro1 = pro1.unsqueeze(1).unsqueeze(2).expand_as(torch.randn(dim1, dim2, 1))
        bert_enc = torch.cat([bert_enc, ofs1, pro1], dim=2)

        logits = self.fc(bert_enc)
        tensor_02 = logits.narrow(1, 0, 1) #缩小张量，将第dim维由start位置处开始取len长的张量
        logits =  tensor_02.squeeze()
        """



        return logits

    #关键方法
    def forward(self, data_x, bert_mask, data_y, offset1, pro1):
        logits = self.compute_logits(data_x, bert_mask, offset1, pro1)
        loss_fct = MultiLabelSoftMarginLoss() #多标签分类损失函数
        loss = loss_fct(logits, data_y)
        return loss

#显著性嵌入的多标签分类模型
class SentenceClassification1(nn.Module):
    def __init__(self, bert_dir, y_num, type_rep):  # 构造函数
        super().__init__()

        self.y_num = y_num
        self.bert = BertModel1.from_pretrained(bert_dir)  # 模型
        self.fc = nn.Linear(self.bert.config.hidden_size + 2, self.y_num)

    def getName(self):
        return self.__class__.__name__

    # logits就是最终的全连接层的输出，神经网络中都是先有logits，而后通过sigmoid函数或者softmax函数得到概率的
    def compute_logits(self, data_x, bert_mask, offset1, pro1,salience):
        outputs = self.bert(data_x, salience,attention_mask=bert_mask)
        bert_enc = outputs[1]  # bert输出的文本表示 这是序列的第一个token(classification token)的最后一层的隐藏状态 hiden_size=768

        ofs1 = offset1.unsqueeze(1)  #位置特征表示
        pro1 = pro1.unsqueeze(1)
        bert_enc = torch.cat([bert_enc, ofs1, pro1], dim=1) #文本表示和位置特征表示按行拼接在一起作为句子表示 hiden_size=770
        logits = self.fc(bert_enc)
        """       
        # 事件类型T的one-hot表示来代替句子级事件标签gs来进行评估，表示相对于事件类型T的单词级显著性
        dim1, dim3 = bert_enc.size()
        dim2 = self.y_num
        bert_enc = bert_enc.unsqueeze(1).expand_as(torch.randn(dim1, dim2, dim3))  # [1,8,768]
        type_rep_x = self.type_rep.unsqueeze(0).expand_as(torch.randn(dim1, dim2, dim3))
        bert_enc = bert_enc + type_rep_x

        # 先代替完在加位置信息
        ofs1 = offset1.unsqueeze(1).unsqueeze(2).expand_as(torch.randn(dim1, dim2, 1))  # [1,8,1]
        pro1 = pro1.unsqueeze(1).unsqueeze(2).expand_as(torch.randn(dim1, dim2, 1))
        bert_enc = torch.cat([bert_enc, ofs1, pro1], dim=2)
        logits = self.fc(bert_enc)
        tensor_02 = logits.narrow(1, 0, 1)  # 缩小张量，将第dim维由start位置处开始取len长的张量
        logits = tensor_02.squeeze()
        """
        
        return logits

    # 关键方法
    def forward(self, data_x, bert_mask, data_y, offset1, pro1,salience):
        logits = self.compute_logits(data_x, bert_mask, offset1, pro1,salience)
        loss_fct = MultiLabelSoftMarginLoss()  # 多标签分类损失函数
        loss = loss_fct(logits, data_y)
        return loss

#显著性嵌入的多分类的模型---单标签
class SentenceClassification2(nn.Module):
    def __init__(self, bert_dir, y_num, type_rep):  # 构造函数
        super().__init__()
        self.y_num = y_num
        self.bert = BertModel1.from_pretrained(bert_dir)  # 模型
        self.fc = nn.Linear(self.bert.config.hidden_size + 2, self.y_num)

    def getName(self):
        return self.__class__.__name__

    # logits就是最终的全连接层的输出，神经网络中都是先有logits，而后通过sigmoid函数或者softmax函数得到概率的
    def compute_logits(self, data_x, bert_mask, offset1, pro1,salience):
        outputs = self.bert(data_x, salience,attention_mask=bert_mask)
        bert_enc = outputs[1]  # bert输出的文本表示 这是序列的第一个token(classification token)的最后一层的隐藏状态 hiden_size=768

        ofs1 = offset1.unsqueeze(1)  #位置特征表示
        pro1 = pro1.unsqueeze(1)
        bert_enc = torch.cat([bert_enc, ofs1, pro1], dim=1) #文本表示和位置特征表示按行拼接在一起作为句子表示 hiden_size=770
        logits = self.fc(bert_enc)
        """
        # 事件类型T的one-hot表示来代替句子级事件标签gs来进行评估，表示相对于事件类型T的单词级显著性
        dim1, dim3 = bert_enc.size()
        dim2 = self.y_num
        bert_enc = bert_enc.unsqueeze(1).expand_as(torch.randn(dim1, dim2, dim3))  # [1,8,768]
        type_rep_x = self.type_rep.unsqueeze(0).expand_as(torch.randn(dim1, dim2, dim3))
        bert_enc = bert_enc + type_rep_x

        # 先代替完在加位置信息
        ofs1 = offset1.unsqueeze(1).unsqueeze(2).expand_as(torch.randn(dim1, dim2, 1))  # [1,8,1]
        pro1 = pro1.unsqueeze(1).unsqueeze(2).expand_as(torch.randn(dim1, dim2, 1))
        bert_enc = torch.cat([bert_enc, ofs1, pro1], dim=2)
        logits = self.fc(bert_enc)
        tensor_02 = logits.narrow(1, 0, 1)  # 缩小张量，将第dim维由start位置处开始取len长的张量
        logits = tensor_02.squeeze()
        """

        return logits

    # 关键方法
    def forward(self, data_x, bert_mask, data_y, offset1, pro1,salience):
        logits = self.compute_logits(data_x, bert_mask, offset1, pro1,salience)
        #换成多分类的损失函数
        loss_fct = CrossEntropyLoss()  # 多分类损失函数
        loss = loss_fct(logits, data_y)
        return loss

#多分类的模型--单标签
class SentenceClassification3(nn.Module):
    def __init__(self, bert_dir, y_num, type_rep):  # 构造函数
        super().__init__()

        self.y_num = y_num
        self.bert = BertModel.from_pretrained(bert_dir)  # 模型
        self.type_rep = type_rep
        self.type_rep.requires_grad = False
        self.fc = nn.Linear(self.bert.config.hidden_size + 2, self.y_num)

    def getName(self):
        return self.__class__.__name__

    # logits就是最终的全连接层的输出，神经网络中都是先有logits，而后通过sigmoid函数或者softmax函数得到概率的
    def compute_logits(self, data_x, bert_mask, offset1, pro1):
        outputs = self.bert(data_x, attention_mask=bert_mask)
        bert_enc = outputs[1]  # bert输出的文本表示 这是序列的第一个token(classification token)的最后一层的隐藏状态 hiden_size=768


        ofs1 = offset1.unsqueeze(1)  #位置特征表示
        pro1 = pro1.unsqueeze(1)
        bert_enc = torch.cat([bert_enc, ofs1, pro1], dim=1) #文本表示和位置特征表示按行拼接在一起作为句子表示 hiden_size=770
        logits = self.fc(bert_enc)

        return logits

    # 关键方法
    def forward(self, data_x, bert_mask, data_y, offset1, pro1):
        logits = self.compute_logits(data_x, bert_mask, offset1, pro1)
        loss_fct = CrossEntropyLoss()  # 多分类损失函数
        loss = loss_fct(logits, data_y)
        return loss

if __name__ == '__main__':


    rep = get_type_rep().detach().cpu().numpy()
    #print(rep)



