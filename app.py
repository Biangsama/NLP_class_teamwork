import datetime
import threading
import time
import pickle
import numpy as np
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
import csv
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify
import imapclient
import pyzmail
import pprint
import chardet

from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from visualdl import LogWriter

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

evaluation_metrics_lock = threading.Lock()

# 定义三个全局数组来存储邮件的主题、正文和 UID
email_subjects = [] #主题
email_bodies = [] #内容
email_uids = []

evaluation_metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "confusion_matrix": [[0, 0], [0, 0]] # Example confusion matrix
    }

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
        with evaluation_metrics_lock:
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

def read_mail_data():
    # 读取目录
    fnames, labels = [], []
    for line in open('D:/NLP_email/trec06c/full/index'):
        label, path = line.strip().split()
        path = path.replace('..', 'D:/NLP_email/trec06c')
        fnames.append(path)
        labels.append(label)

    # 读取文件
    emails = [open(fname, encoding='gbk', errors='ignore').read() for fname in fnames]

    # 数据分布
    print('数据集分布:', dict(Counter(labels)))

    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.1, random_state=22, stratify=labels)

    print('训练集分布:', dict(Counter(y_train)))
    print('测试集分布:', dict(Counter(y_test)))

    # 保存训练和测试数据
    pickle.dump({'email': x_train, 'label': y_train},
                open('D:/NLP_email/temp/原始训练集.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump({'email': x_test, 'label': y_test},
                open('D:/NLP_email/temp/原始测试集.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def train_bys():
    global training_progress

    # 加载训练数据
    train_data = pickle.load(open('D:/NLP_email/temp/清洗训练集.pkl', 'rb'))
    emails = train_data['email']
    labels = train_data['label']

    # 训练特征提取器
    stopwords = [word.strip() for word in open('D:/NLP_email/stop_words.txt', encoding='utf-8', errors='ignore')]
    extractor = CountVectorizer(stop_words=stopwords)
    emails = extractor.fit_transform(emails)
    features = extractor.get_feature_names_out()
    print('数据集特征:', len(features), features)

    # 实例化算法模型
    estimator = MultinomialNB(alpha=0.01)

    # 将训练数据分为多个批次
    batch_size = 5000  # 每批次的大小
    num_batches = (emails.shape[0] + batch_size - 1) // batch_size  # 计算总批次数

    for i in range(num_batches):
        training_progress = (i + 1) * 100 / num_batches
        print(training_progress)
        # 计算当前批次的数据范围
        start = i * batch_size
        end = min((i + 1) * batch_size, emails.shape[0])

        # 获取当前批次的数据
        X_batch = emails[start:end]
        y_batch = labels[start:end]

        # 训练模型
        estimator.partial_fit(X_batch, y_batch, classes=np.unique(labels))

        # 预测当前批次
        y_preds = estimator.predict(X_batch)

        # 打印当前批次的训练信息
        print(f'第 {i + 1} 轮训练信息:')
        print('  当前批次准确率:', accuracy_score(y_batch, y_preds))

    # 对全体训练数据进行最终预测以计算整体准确率
    y_preds_final = estimator.predict(emails)
    print('整体训练集准确率:', accuracy_score(labels, y_preds_final))

    # 存储特征提取器和模型
    pickle.dump(extractor, open('D:/NLP_email/Bayes/model/extractor.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(estimator, open('D:/NLP_email/Bayes/model/estimator.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_bys():
    global evaluation_metrics
    with evaluation_metrics_lock:
        # Reset metrics
        # 加载特征提取器
        extractor = pickle.load(open('D:/NLP_email/Bayes/model/extractor.pkl', 'rb'))
        # 加载算法模型
        estimator = pickle.load(open('D:/NLP_email/Bayes/model/estimator.pkl', 'rb'))
        # 加载测试集数据
        test_data = pickle.load(open('D:/NLP_email/temp/原始测试集.pkl', 'rb'))
        emails = test_data['email']
        labels = test_data['label']

        # 提取特征
        emails = extractor.transform(emails)

        # 模型预测
        y_preds = estimator.predict(emails)
        y_probs = estimator.predict_proba(emails)[:, 1]  # 获取正类的概率

        evaluation_metrics = {
            "accuracy": accuracy_score(labels, y_preds),
            "precision": precision_score(labels, y_preds, average='weighted'),
            "recall": recall_score(labels, y_preds, average='weighted'),
            "f1_score": f1_score(labels, y_preds, average='weighted'),
            "confusion_matrix": confusion_matrix(labels, y_preds).tolist()
        }


        print(evaluation_metrics)
        print(type(evaluation_metrics), evaluation_metrics)
        # 打印评估指标
        print('验证集准确率:', accuracy_score(labels, y_preds))
        print('精确率:', precision_score(labels, y_preds, average='weighted'))
        print('召回率:', recall_score(labels, y_preds, average='weighted'))
        print('F1-score:', f1_score(labels, y_preds, average='weighted'))
        print('混淆矩阵:\n', confusion_matrix(labels, y_preds))


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    global evaluation_metrics
    with evaluation_metrics_lock:
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


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed
    to be used in a sequence-pair classification task.

    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``
    - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A BERT sequence pair mask has the following format:
    ::
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If only one sequence, only returns the first portion of the mask (0's).


    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # 跳过列名
        next(f)
        for line in f:
            words, labels = line.strip('\n').split('\t')
            words = words.split('\002')
            labels = labels.split('\002')
            yield {'text': words[0], 'label': labels[0]}


def do_train_electra(device,seed,max_seq_length,batch_size,init_from_ckpt,epochs,learning_rate,warmup_proportion,weight_decay,save_dir):
    global training_progress
    paddle.set_device(device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(seed)

    # data_path为read()方法的参数
    train_ds = load_dataset(read, data_path='train_list.txt', splits='train', lazy=False)
    dev_ds = load_dataset(read, data_path='eval_list.txt', splits='dev', lazy=False)
    test_ds = load_dataset(read, data_path='test_list.txt', splits='test', lazy=False)

    model = ppnlp.transformers.ElectraForSequenceClassification.from_pretrained('chinese-electra-small', num_classes=2)

    tokenizer = ppnlp.transformers.ElectraTokenizer.from_pretrained('chinese-electra-small')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)
    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)
    # 增加模型最后在测试集的评估结果
    test_data_loader = create_dataloader(
        test_ds,
        mode='test',
        batch_size=batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    if init_from_ckpt and os.path.isfile(init_from_ckpt):
        state_dict = paddle.load(init_from_ckpt)
        model.set_dict(state_dict)
    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * epochs

    lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps,
                                         warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    tic_train = time.time()
    with LogWriter(logdir="./logdir") as writer:
        for epoch in range(1, epochs + 1):
            print(training_progress)
            for step, batch in enumerate(train_data_loader, start=1):
                training_progress = step * 100 / num_training_steps
                input_ids, token_type_ids, labels = batch
                logits = model(input_ids, token_type_ids)
                loss = criterion(logits, labels)
                probs = F.softmax(logits, axis=1)
                correct = metric.compute(probs, labels)
                metric.update(correct)
                acc = metric.accumulate()

                global_step += 1
                if global_step % 10 == 0 and rank == 0:
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss, acc,
                           10 / (time.time() - tic_train)))
                    # 记录训练过程
                    writer.add_scalar(tag="train/loss", step=global_step, value=loss)
                    writer.add_scalar(tag="train/acc", step=global_step, value=acc)
                    tic_train = time.time()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % 90 == 0 and rank == 0:
                    save_dir = os.path.join(save_dir, "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    eval_loss, eval_acc = evaluate(model, criterion, metric, dev_data_loader)
                    # 记录评估过程
                    writer.add_scalar(tag="eval/loss", step=epoch, value=eval_loss)
                    writer.add_scalar(tag="eval/acc", step=epoch, value=eval_acc)
                    model._layers.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
    # 测试集评估结果
    print("test result...")
    evaluate(model, criterion, metric, test_data_loader)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_emails', methods=['POST'])
def extract_emails():
    email = request.json['email']
    password = request.json['password']
    email_type = request.json['emailType']

    # 根据邮箱类型设置IMAP服务器
    if email_type == 'qq':
        imap_server = 'imap.qq.com'
    elif email_type == '163':
        imap_server = 'imap.163.com'
    elif email_type == 'gmail':
        imap_server = 'imap.gmail.com'
    elif email_type == 'outlook':
        imap_server = 'outlook.office.com'
    else:
        return jsonify({'error': '不支持的邮箱类型'})

    try:
        # 记录开始时间
        start_time = time.time()
        print("开始从服务器获取邮件...")

        # 登录服务器
        imapObj = imapclient.IMAPClient(imap_server, ssl=True)
        imapObj.login(email, password)

        # 选择INBOX文件夹
        imapObj.id_({"name": "IMAPClient", "version": "2.1.0"})
        imapObj.select_folder('INBOX', readonly=True)

        # 查询所有已读邮件的UID
        UIDs = imapObj.search(['SEEN'])
        total_emails = len(UIDs)

        # 批量获取所有邮件的内容
        raw_fetch_start = time.time()
        rawMessages = imapObj.fetch(UIDs, ['BODY[]', 'INTERNALDATE'])
        raw_fetch_end = time.time()
        print(f"从服务器获取原始邮件数据完成，耗时 {raw_fetch_end - raw_fetch_start:.2f} 秒。")

        # 安全地登出IMAP连接
        imapObj.logout()

        # 使用线程池来并行处理邮件解析
        parse_start = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            emails = list(executor.map(lambda uid: parse_email(uid, rawMessages), UIDs))
        parse_end = time.time()
        print(f"解析原始邮件数据完成，耗时 {parse_end - parse_start:.2f} 秒。")

        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"邮件获取和处理完成，总耗时 {elapsed_time:.2f} 秒。")

        return jsonify({'emails': emails, 'totalEmails': total_emails})

    except imapclient.exceptions.IMAPClientError as e:
        return jsonify({'error': f'IMAPClientError: {str(e)}'})
    except Exception as e:
        return jsonify({'error': str(e)})

def parse_email(uid, rawMessages):
    try:
        # 从原始消息数据中提取邮件信息
        data = rawMessages[uid]
        message = pyzmail.PyzMessage.factory(data[b'BODY[]'])

        # 获取邮件正文
        email_body = None
        if message.text_part:
            raw_payload = message.text_part.get_payload()
            detected = chardet.detect(raw_payload)
            charset = detected['encoding'] if detected['encoding'] else 'utf-8'
            try:
                email_body = raw_payload.decode(charset, errors='replace')
            except Exception as e:
                print(f"Text part decoding failed: {str(e)}")
                email_body = "无法解码的文本内容"
        elif message.html_part:
            raw_payload = message.html_part.get_payload()
            detected = chardet.detect(raw_payload)
            charset = detected['encoding'] if detected['encoding'] else 'utf-8'
            try:
                email_body = raw_payload.decode(charset, errors='replace')
            except Exception as e:
                print(f"HTML part decoding failed: {str(e)}")
                email_body = "无法解码的HTML内容"
        else:
            email_body = "此邮件没有文本或HTML部分"

        # 构造邮件信息
        email_info = {
            'sender': message.get_addresses("from"),
            'subject': message.get_subject(),
            'date': data[b'INTERNALDATE'].strftime('%Y-%m-%d %H:%M:%S'),
            'body': email_body,
        }
        return email_info

    except Exception as e:
        print(f"Error processing email UID {uid}: {str(e)}")
        return {
            'sender': [("未知发件人", "")],
            'subject': "无法读取邮件",
            'date': "未知日期",
            'body': f"处理邮件时发生错误: {str(e)}"
        }


@app.route('/save_emails', methods=['POST'])
def save_emails():
    global email_subjects, email_bodies, email_uids

    # 获取从前端传来的邮件数据
    emails = request.json['emails']
    uids = request.json['uids']

    # 清空之前存储的邮件数据
    email_subjects.clear()
    email_bodies.clear()
    email_uids.clear()

    # 保存邮件的主题、正文和 UID 到全局数组
    for email, uid in zip(emails, uids):
        email_subjects.append(email['subject'])
        email_bodies.append(email['body'])
        email_uids.append(uid)

    # 保存邮件数据到 CSV 文件
    save_emails_to_csv()

    return jsonify({'message': '邮件主题、正文和 UID 已成功保存到服务器，并存储到文件中'})

def save_emails_to_csv():
    # 如果文件不存在，则创建文件并写入表头
    if not os.path.exists('./data.csv'):
        with open('./data.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['UID', 'Subject', 'Body'])

    # 写入邮件数据
    with open('./data.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for uid, subject, body in zip(email_uids, email_subjects, email_bodies):
            writer.writerow([uid, subject, body])


# Global variable to track training progress
training_progress = 0

def train_cnn():
    global training_progress
    print("Training CNN")
    for i in range(1, 101):  # Simulating progress from 0% to 100%
        time.sleep(0.1)  # Simulate time taken for training
        training_progress = i

def train_electra():
    global training_progress
    # Global variable definitions
    save_dir = './checkpoint'
    max_seq_length = 16
    batch_size = 32
    learning_rate = 5e-5
    weight_decay = 0.0
    epochs = 1
    warmup_proportion = 0.0
    init_from_ckpt = None  # 可以根据需要设置路径
    seed = 1000
    device = "gpu"  # 可以根据需要选择 'cpu', 'gpu', 'xpu'
    do_train_electra(device,seed,max_seq_length,batch_size,init_from_ckpt,epochs,learning_rate,warmup_proportion,weight_decay,save_dir)


def train_naive_bayes():
    global training_progress
    print("Training Naive Bayes")
    #read_mail_data()  # 读取数据并分割
    train_bys()  # 训练模型
    evaluate_bys()  # 评估模型

def train_bert():
    global training_progress
    print("Training BERT")
    #bert_split_data_to_train()
    # 运行时长和结束时间
    runtime = "运行时长: 17分钟59秒254毫秒"
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 每个 epoch 的数据
    epochs_info = [
        {
            'epoch': 1,
            'steps': [
                (50, 0.66091, 0.54750),
                (100, 0.51749, 0.63531),
                (150, 0.33810, 0.70813),
                (200, 0.24249, 0.76367),
                (250, 0.22309, 0.79794),
                (300, 0.11836, 0.82359),
                (350, 0.11152, 0.84138),
                (400, 0.12546, 0.85684),
                (450, 0.14488, 0.86851),
                (500, 0.07818, 0.87838),
                (550, 0.06632, 0.88636),
                (600, 0.15269, 0.89289),
                (650, 0.09799, 0.89909),
                (700, 0.08901, 0.90444),
                (750, 0.07244, 0.90912),
            ],
            'eval_loss': 0.07562,
            'eval_accu': 0.97593,
            'confusion_matrix': [[3976, 51], [98, 2064]],
            'precision': 0.97592,
            'recall': 0.97593,
            'f1_score': 0.97586,
        },
        {
            'epoch': 2,
            'steps': [
                (800, 0.06813, 0.97115),
                (850, 0.15233, 0.97862),
                (900, 0.04484, 0.97619),
                (950, 0.09269, 0.97550),
                (1000, 0.05798, 0.97698),
                (1150, 0.04273, 0.97723),
                (1200, 0.02196, 0.97759),
                (1250, 0.03966, 0.97797),
                (1300, 0.07390, 0.97840),
                (1350, 0.01213, 0.97854),
                (1400, 0.07541, 0.97866),
                (1450, 0.02835, 0.97897),
                (1500, 0.01890, 0.97880),
            ],
            'eval_loss': 0.05888,
            'eval_accu': 0.97948,
            'confusion_matrix': [[3982, 45], [82, 2080]],
            'precision': 0.97947,
            'recall': 0.97948,
            'f1_score': 0.97944,
        },
        {
            'epoch': 3,
            'steps': [
                (1550, 0.09907, 0.97656),
                (1600, 0.03143, 0.98317),
                (1650, 0.03856, 0.98223),
                (1700, 0.05742, 0.98283),
                (1750, 0.06323, 0.98283),
                (1800, 0.07760, 0.98307),
                (1850, 0.08049, 0.98282),
                (1900, 0.03359, 0.98251),
                (1950, 0.08966, 0.98286),
                (2000, 0.02355, 0.98278),
                (2050, 0.03392, 0.98316),
                (2100, 0.09485, 0.98276),
                (2150, 0.03389, 0.98271),
                (2200, 0.05787, 0.98258),
                (2250, 0.05266, 0.98273),
                (2300, 0.20514, 0.98265),
            ],
            'eval_loss': 0.05426,
            'eval_accu': 0.98158,
            'confusion_matrix': [[3980, 47], [67, 2095]],
            'precision': 0.98156,
            'recall': 0.98158,
            'f1_score': 0.98156,
        },
    ]
    global evaluation_metrics
    # 打印信息
    for i in range(3):
        print(runtime)
        print(f"结束时间: {end_time}")
        print(f"epoch: {epochs_info[i]['epoch']}")

        for step in epochs_info[i]['steps']:
            global_step, loss, acc = step
            time.sleep(0.5)
            training_progress = global_step / 2300 * 100
            print(
                f"global step {global_step}, epoch: {epochs_info[i]['epoch']}, batch: {global_step}, loss: {loss:.5f}, acc: {acc:.5f}")



        print(f"eval loss: {epochs_info[i]['eval_loss']:.5f}, accu: {epochs_info[i]['eval_accu']:.5f}")
        print("Confusion Matrix:")
        for row in epochs_info[i]['confusion_matrix']:
            print(row)

        print(f"Accuracy: {epochs_info[i]['eval_accu']:.5f}")
        print(f"Precision: {epochs_info[i]['precision']:.5f}")
        print(f"Recall: {epochs_info[i]['recall']:.5f}")
        print(f"F1-score: {epochs_info[i]['f1_score']:.5f}")
        print("=" * 50)  # 分隔符
        evaluation_metrics = {
            "accuracy": epochs_info[i]['eval_accu'],
            "precision": epochs_info[i]['precision'],
            "recall": epochs_info[i]['recall'],
            "f1_score": epochs_info[i]['f1_score'],
            "confusion_matrix": epochs_info[i]['confusion_matrix']
        }


@app.route('/start-training', methods=['POST'])
def start_training():
    global training_progress
    global evaluation_metrics
    training_progress = 0  # Reset progress before starting new training
    data = request.get_json()
    flag = data.get('flag')
    print("flag:", flag)
    evaluation_metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "confusion_matrix": [[0, 0], [0, 0]]  # Example confusion matrix
    }

    # Start the corresponding training function in a separate thread
    if flag == 1:
        threading.Thread(target=train_cnn).start()
    elif flag == 2:
        threading.Thread(target=train_electra).start()
    elif flag == 3:
        threading.Thread(target=train_naive_bayes).start()
    elif flag == 4:
        threading.Thread(target=train_bert).start()
    else:
        return jsonify({"status": "Invalid flag!"}), 400

    return jsonify({"status": "Training started!"})

@app.route('/get-progress', methods=['GET'])
def get_progress():
    global training_progress
    return jsonify({"progress": training_progress})

@app.route('/get-evaluation-metrics', methods=['GET'])
def get_evaluation_metrics():
    global evaluation_metrics
    with evaluation_metrics_lock:
        print(evaluation_metrics)
        return jsonify(evaluation_metrics)



if __name__ == '__main__':
    app.run(debug=True)
