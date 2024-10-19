# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# 其他代码逻辑可以在这里继续使用这些全局变量



def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
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
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

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

def do_train():
    paddle.set_device(device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(seed)

    # train_ds, dev_ds = load_dataset("chnsenticorp", splits=["train", "dev"])

    # data_path为read()方法的参数
    train_ds = load_dataset(read,data_path='/home/aistudio/train_list.txt',splits='train',lazy=False)
    dev_ds = load_dataset(read,data_path='/home/aistudio/eval_list.txt',splits='dev',lazy=False)
    test_ds = load_dataset(read,data_path='/home/aistudio/test_list.txt',splits='test',lazy=False)

    # If you wanna use bert/roberta/electra pretrained model,
    # model = ppnlp.transformers.BertForSequenceClassification.from_pretrained('bert-base-chinese', num_class=2)
    #model = ppnlp.transformers.RobertaForSequenceClassification.from_pretrained('rbt3', num_class=2)
    model = ppnlp.transformers.ElectraForSequenceClassification.from_pretrained('chinese-electra-small', num_classes=2)
    # model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained('ernie-tiny', num_classes=len(train_ds.label_list))

    # If you wanna use bert/roberta/electra pretrained model,
    # tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained('bert-base-chinese')
    #tokenizer = ppnlp.transformers.RobertaTokenizer.from_pretrained('rbt3')
    tokenizer = ppnlp.transformers.ElectraTokenizer.from_pretrained('chinese-electra-small')
    # ErnieTinyTokenizer is special for ernie-tiny pretained model.
    # tokenizer = ppnlp.transformers.ErnieTinyTokenizer.from_pretrained('ernie-tiny')

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
            for step, batch in enumerate(train_data_loader, start=1):
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
                    #记录训练过程
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
                    #记录评估过程
                    writer.add_scalar(tag="eval/loss", step=epoch, value=eval_loss)
                    writer.add_scalar(tag="eval/acc", step=epoch, value=eval_acc)
                    model._layers.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
    # 测试集评估结果
    print("test result...")
    evaluate(model, criterion, metric, test_data_loader)

if __name__ == "__main__":
    do_train()
