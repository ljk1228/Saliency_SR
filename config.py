bert_dir = './data/BertModel/bert-base-uncased' #uncased不区分大小写
#bert_dir = './data/BertModel/bert-large-cased'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(bert_dir, do_lower_case=False)

step_type=['RST','CTN','BAC','PUR','GAP','MTD','IMP','CLN']

#print("语步类型的长度",len(step_type))
tag2idx = {tag: idx for idx, tag in enumerate(step_type)}
idx2tag = {idx: tag for idx, tag in enumerate(step_type)}
print(tag2idx)
print(idx2tag)



