import json
import pickle
import config_step as config
import pandas as pd
import ast
import numpy
import pickle
tokenizer = config.tokenizer


def build_examples(data):#data是dataframe
    res = []
    for index, row in data.iterrows():
        text,text_copy,offset1,pro1,labels=row['text'],row['text_copy'],row['offset1'],row['pro1'],row['labels']
        labels = ast.literal_eval(labels)#字符串列表转为列表

        sentence = text.split() #text是拼接后的句子,text_copy是原始句子 [word,word,word,word,....]
        words = ['[CLS]'] + sentence + ['[SEP]']

        subword_ids = list()
        spans = list()

        for word in words:
            sub_tokens = tokenizer.tokenize(word)
            sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)

            s = len(subword_ids)
            subword_ids.extend(sub_tokens)
            e = len(subword_ids) - 1
            spans.append([s, e])

        res.append([subword_ids, spans, labels,text_copy,offset1,pro1,words])
    return res



if __name__ == '__main__':


    df = pd.read_csv('./data/STEP/multi_l_r_new_all_results.csv')
    # 随机打乱数据集
    df = df.sample(frac=1, random_state=1)
    bound = int(0.8 * len(df))

    train_df = df[:bound]
    eval_df = df[bound:]
    print(f'train_df has length: {len(train_df)}')
    print(f'eval_df has length: {len(eval_df)}')


    data = {}
    data['train'] = build_examples(train_df)
    data['val'] = build_examples(eval_df)
    data['test'] = data['val']
    print(data['train'][0])


    f = open('data/data_step.pk', 'wb')
    pickle.dump(data, f)
    f.close()



