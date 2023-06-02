import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pickle
import torch

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

import config_step as config
from dataset_step import Dataset
from model_step import SentenceClassification, get_type_rep
from utils import save_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
def load_dataset():
    filename = 'data/data_step.pk'
    data = pickle.load(open(filename, 'rb'))
    return data['train'], data['val'], data['test']


def evaluate(model, device, testset):

    #评估部分有预测
    golds = []
    predicts = []
    for batch in testset.get_tqdm(device, False):
        data_x, data_x_mask, data_labels, data_span, offset1, pro1, appendix = batch
        logits = model.compute_logits(data_x, data_x_mask, offset1, pro1)
        logits = torch.nn.Sigmoid()(logits)
        golds.extend(data_labels.detach().cpu().numpy())
        predicts.extend(logits.detach().cpu().numpy())

    label_list = ['RST','CTN','BAC','PUR','GAP','MTD','IMP','CLN']
    accuracy = metrics.accuracy_score(golds, (np.array(predicts) > 0.3).astype(int))
    f1_score_micro = metrics.f1_score(golds, (np.array(predicts) > 0.3).astype(int), average='micro')
    print(f"Accuracy Score = {accuracy}",f"F1 Score (Micro) = {f1_score_micro}")
    print(classification_report(golds, (np.array(predicts) > 0.3).astype(int), target_names=label_list,digits=4))

def evaluate1(model, device, testset):
    golds = []
    predicts = []

    for batch in testset.get_tqdm(device, False):
        data_x, data_x_mask, data_labels, data_span, offset1, pro1, appendix = batch
        logits = model.compute_logits(data_x, data_x_mask, offset1, pro1)
        logits = torch.nn.Sigmoid()(logits)
        golds.extend(data_labels.detach().cpu().numpy())
        predicts.extend(logits.detach().cpu().numpy())


    n_gold, n_predict, n_correct = 0, 0, 0
    th = 0.3
    for g, p in zip(golds, predicts):
        n_gold += sum(g)
        n_predict += sum(p >= th)  # 预测为正
        n_correct += sum(g * (p > th))  # 正确预测为正

    p = n_correct / n_predict
    r = n_correct / n_gold
    f1 = 2 * p * r / (p + r)
    print(n_gold, n_predict, n_correct, "精确度：", p, "召回率：", r, "f1值：", f1)
if __name__ == '__main__':

    device = 'cuda'

    lr = 1e-5
    batch_size = 16
    n_epochs = 5

    train_data, val_data, test_data = load_dataset()
    train_dataset =Dataset(batch_size, 512, train_data)
    test_dataset = Dataset(batch_size, 512, test_data)
    #[data_x, data_x_mask, data_labels, data_span, offset1, pro1, appendix]

    rep = get_type_rep().detach_().to(device)

    model = SentenceClassification(config.bert_dir, len(config.idx2tag),rep)
    model.to(device)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(parameters, lr=lr, correct_bias=False)

    for idx in range(n_epochs):
        model.train()
        for batch in train_dataset.get_tqdm(device, True):
            data_x, data_x_mask, data_labels, data_span, offset1, pro1, appendix = batch

            #在这里拼到输入到model
            loss = model(data_x, data_x_mask, data_labels,offset1, pro1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.zero_grad()
        #用训练集数据训练一个句子级别的分类器
        save_model(model, 'model/step_sentence_shao%d.ckp' % (idx))



        model.eval()
        evaluate(model, device, test_dataset)

