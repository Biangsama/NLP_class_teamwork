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
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class SelfDefinedDataset(paddle.io.Dataset):
    def __init__(self, data):
        super(SelfDefinedDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return ["0", "1"]


def txt_to_list(file_name):
    res_list = []
    for line in open(file_name, encoding='utf-8'):
        res_list.append(line.strip().split('\t'))
    return res_list


def bert_split_data_to_train():
    trainlst = txt_to_list('train_list_2.txt')
    devlst = txt_to_list('eval_list_2.txt')
    testlst = txt_to_list('test_list_2.txt')

    train_ds = SelfDefinedDataset(trainlst)
    dev_ds = SelfDefinedDataset(devlst)
    test_ds = SelfDefinedDataset(testlst)
    label_list = train_ds.get_labels()
    print(label_list)
    from paddlenlp.datasets import MapDataset

    train_ds = MapDataset(train_ds)
    dev_ds = MapDataset(dev_ds)
    test_ds = MapDataset(test_ds)
    print("训练集数据：{}\n".format(train_ds[0:3]))
    print("验证集数据:{}\n".format(dev_ds[0:3]))
    print("测试集数据:{}\n".format(test_ds[0:3]))
    # 看看数据长什么样子，分别打印训练集、验证集、测试集的前3条数据。

    print("训练集样本个数:{}".format(len(train_ds)))
    print("验证集样本个数:{}".format(len(dev_ds)))
    print("测试集样本个数:{}".format(len(test_ds)))

    def convert_example(example, tokenizer, label_list, max_seq_length=256, is_test=False):
        if is_test:
            text = example
        else:
            text, label = example
        # tokenizer.encode方法能够完成切分token，映射token ID以及拼接特殊token
        encoded_inputs = tokenizer.encode(text=text, max_seq_len=max_seq_length)
        input_ids = encoded_inputs["input_ids"]
        # 注意，在早前的PaddleNLP版本中，token_type_ids叫做segment_ids
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
    # 调用ppnlp.transformers.BertTokenizer进行数据处理，tokenizer可以把原始输入文本转化成模型model可接受的输入数据格式。
    tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained("bert-base-chinese",truncation=True)
    # 使用partial()来固定convert_example函数的tokenizer, label_list, max_seq_length, is_test等参数值
    trans_fn = partial(convert_example, tokenizer=tokenizer, label_list=label_list, max_seq_length=128, is_test=False)
    batchify_fn = lambda samples, fn=Tuple(Pad(axis=0, pad_val=tokenizer.pad_token_id),
                                           Pad(axis=0, pad_val=tokenizer.pad_token_id), Stack(dtype="int64")): [data for
                                                                                                                data in
                                                                                                                fn(samples)]
    # 训练集迭代器
    train_loader = create_dataloader_bert(train_ds, mode='train', batch_size=64, batchify_fn=batchify_fn, trans_fn=trans_fn)
    # 验证集迭代器
    dev_loader = create_dataloader_bert(dev_ds, mode='dev', batch_size=64, batchify_fn=batchify_fn, trans_fn=trans_fn)
    # 测试集迭代器
    test_loader = create_dataloader_bert(test_ds, mode='test', batch_size=64, batchify_fn=batchify_fn, trans_fn=trans_fn)

    model = ppnlp.transformers.BertForSequenceClassification.from_pretrained("bert-base-chinese", num_classes=2)

    # 设置训练超参数

    # 学习率
    learning_rate = 1e-6
    # 训练轮次
    epochs = 3
    # 学习率预热比率
    warmup_proption = 0.1
    # 权重衰减系数
    weight_decay = 0.01

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(warmup_proption * num_training_steps)

    def get_lr_factor(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return max(0.0,
                       float(num_training_steps - current_step) /
                       float(max(1, num_training_steps - num_warmup_steps)))

    # 学习率调度器
    lr_scheduler = paddle.optimizer.lr.LambdaDecay(learning_rate,
                                                   lr_lambda=lambda current_step: get_lr_factor(current_step))

    # 优化器
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    # 损失函数
    criterion = paddle.nn.loss.CrossEntropyLoss()
    # 评估函数
    metric = paddle.metric.Accuracy()

    # 评估函数，设置返回值，便于VisualDL记录
    def evaluate(model, criterion, metric, data_loader):
        global evaluation_metrics
        model.eval()
        metric.reset()
        losses = []
        all_preds = []
        all_labels = []

        for batch in data_loader:
            input_ids, token_type_ids, labels = batch
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            losses.append(loss.numpy())

            probs = F.softmax(logits, axis=1)
            preds = paddle.argmax(probs, axis=1).numpy()  # 获取预测的标签

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())  # 记录真实标签

            correct = metric.compute(probs, labels)
            metric.update(correct)
            accu = metric.accumulate()

        # 计算评估指标
        cm = confusion_matrix(all_labels, all_preds).tolist()
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        evaluation_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }

        print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
        print("Confusion Matrix:\n", cm)
        print("Accuracy: %.5f" % accuracy)
        print("Precision: %.5f" % precision)
        print("Recall: %.5f" % recall)
        print("F1-score: %.5f" % f1)

        model.train()
        metric.reset()
        return np.mean(losses), accu

    global_step = 0
    with LogWriter(logdir="./log") as writer:
        for epoch in range(1, epochs + 1):
            print("epoch:", epoch)
            for step, batch in enumerate(train_loader, start=1):  # 从训练数据迭代器中取数据
                print("step:", step)
                input_ids, segment_ids, labels = batch
                #print("input_ids:", input_ids)
                logits = model(input_ids, segment_ids)
                #print("logits:", logits)
                loss = criterion(logits, labels)  # 计算损失
                print("loss:", loss)
                probs = F.softmax(logits, axis=1)
                correct = metric.compute(probs, labels)
                metric.update(correct)
                acc = metric.accumulate()
                print("acc:", acc)

                global_step += 1
                print('global_step:', global_step)
                if global_step % 50 == 0:
                    print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (
                    global_step, epoch, step, loss, acc))
                    # 记录训练过程
                    writer.add_scalar(tag="train/loss", step=global_step, value=loss)
                    writer.add_scalar(tag="train/acc", step=global_step, value=acc)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_gradients()
            eval_loss, eval_acc = evaluate(model, criterion, metric, dev_loader)
            # eval_loss2, eval_acc2 = evaluate(model, criterion, metric, dev_loader2)
            # 在每个epoch后保存模型
            paddle.save(model.state_dict(), f'./saved_models/model_epoch_test{epoch}.pdparams')
            # 记录评估过程
            writer.add_scalar(tag="eval/loss", step=epoch, value=eval_loss)
            writer.add_scalar(tag="eval/acc", step=epoch, value=eval_acc)
            # writer.add_scalar(tag="eval/loss2", step=epoch, value=eval_loss2)
            # writer.add_scalar(tag="eval/acc2", step=epoch, value=eval_acc2)



#数据迭代器构造方法
def create_dataloader_bert(dataset, trans_fn=None, mode='train', batch_size=1, use_gpu=True, pad_token_id=0, batchify_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn, lazy=True)

    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        shuffle = True if mode == 'train' else False #如果不是训练集，则不打乱顺序
        sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle) #生成一个取样器
    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, return_list=True, collate_fn=batchify_fn)
    return dataloader


#数据预处理

bert_split_data_to_train()
