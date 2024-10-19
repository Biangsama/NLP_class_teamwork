from translate import Translator
# 导入相关的模块
import re
import jieba
import os
import random
import paddle
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Pad, Tuple
import paddle.nn.functional as F
import paddle.nn as nn
from visualdl import LogWriter
import numpy as np
from functools import partial #partial()函数可以用来固定某些参数值，并返回一个新的callable对象

#调用ppnlp.transformers.BertTokenizer进行数据处理，tokenizer可以把原始输入文本转化成模型model可接受的输入数据格式。
tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained("bert-base-chinese")

translator=Translator(from_lang="english",to_lang="chinese")
# translation = translator.translate("can you speak english?")
# print(translation)
def translate_text(text):
    try:
        translator = Translator(to_lang="zh")
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        return None

#数据预处理
def convert_example(example,tokenizer,label_list,max_seq_length=256,is_test=False):
    if is_test:
        text = example
    else:
        text, label = example
    #tokenizer.encode方法能够完成切分token，映射token ID以及拼接特殊token
    encoded_inputs = tokenizer.encode(text=text, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    #注意，在早前的PaddleNLP版本中，token_type_ids叫做segment_ids
    segment_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label_map = {}
        for (i, l) in enumerate(label_list):
            label_map[l] = i

        label = label_map[label]
        label = np.array([label], dtype="int64")
        return input_ids, segment_ids, label
    else:
        return input_ids, segment_ids


#用于加载训练模型权重验证
model2 = ppnlp.transformers.BertForSequenceClassification.from_pretrained("bert-base-chinese", num_classes=2)



def predict(model, data, tokenizer, label_map, batch_size=1):
    examples = []
    for text in data:
        input_ids, segment_ids = convert_example(text, tokenizer, label_list=label_map.values(),  max_seq_length=128, is_test=True)
        examples.append((input_ids, segment_ids))

    batchify_fn = lambda samples, fn=Tuple(Pad(axis=0, pad_val=tokenizer.pad_token_id), Pad(axis=0, pad_val=tokenizer.pad_token_id)): fn(samples)
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in batches:
        input_ids, segment_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        logits = model(input_ids, segment_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results

params_path = './saved_models/model_epoch3.pdparams'
state_dict = paddle.load(params_path)
model2.set_state_dict(state_dict)
data = ['您好我公司有多余的发票可以向外代开,国税,地税,运输,广告,海关缴款书如果贵公司,厂,有需要请来电洽谈,咨询联系电话,罗先生谢谢顺祝商祺']
label_map = {0: '垃圾邮件', 1: '正常邮件'}
data1=translate_text("It doesn't look like there's much interest in a bof;maybe next time.")
data2=translate_text("if you want a bof, schedule it and see what happens.i'm a little curious myself.")
data=['老师好，请问一下这周需要上课吗',data1,data2]
predictions = predict(model2, data, tokenizer, label_map, batch_size=32)
for idx, text in enumerate(data):
    print('预测内容: {} \n邮件标签: {}'.format(text, predictions[idx]))