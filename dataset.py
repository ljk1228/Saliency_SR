import copy
import sys
import torch
import pickle
import random
import numpy as np

from tqdm import tqdm


class Dataset(object):
    def __init__(self, batch_size, seq_len, dataset):
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.construct_index(dataset)


    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size,
                    leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):

        #subword_ids,spans, labels,text,offset1,pro1,words

        data_x, data_x_mask, data_labels, data_span, offset1, pro1, appendix = list(), list(), list(),list(), list(), list(), list()

        #长度统一data_x 和data_x_mask
        max_sentence_len = self.seq_len
        for data in batch:
            sentence_lens=len(data[0])
            data_span_lens = len(data[1])
            input_mask = [1] *  sentence_lens
            if sentence_lens > max_sentence_len:  # notice # 超过则截断
                data[0] = data[0][:max_sentence_len]
                input_mask=input_mask[:max_sentence_len]
            # Zero-pad up to the sequence length.
            while sentence_lens < max_sentence_len:
                data[0].append(0)
                input_mask.append(0)
                sentence_lens+=1

            if data_span_lens > max_sentence_len:  # data_span长度目前和input_idx统一
                data[1]=data[1][:max_sentence_len]
            while data_span_lens < max_sentence_len:
                data[1].append([0,0])
                data_span_lens += 1

            #data_span 长度目前和input_mask统一
            data_x.append(data[0])
            data_x_mask.append(input_mask)
            data_labels.append(data[2])
            data_span.append(data[1])
            offset1.append(data[4])
            pro1.append(data[5])
            appendix.append(data[6])



        f = torch.LongTensor

        data_x = f(data_x)
        data_x_mask = f(data_x_mask)
        data_labels = f(data_labels)
        data_span = f(data_span)
        offset1 = f(offset1)
        pro1 = f(pro1)


        return [data_x.to(device),
                data_x_mask.to(device),
                data_labels.to(device),
                data_span.to(device),
                offset1.to(device),
                pro1.to(device),
                appendix]



