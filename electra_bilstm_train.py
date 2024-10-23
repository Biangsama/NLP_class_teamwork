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
from functools import partial
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from paddle.io import Dataset
from paddlenlp.transformers import BertModel


class SelfDefinedDataset(Dataset):
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
        res_list.append(line.strip().split('	'))
    return res_list


def bert_split_data_to_train():
    global train_ds, dev_ds, test_ds
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
    print("训练集样本个数:{}".format(len(train_ds)))
    print("验证集样本个数:{}".format(len(dev_ds)))
    print("测试集样本个数:{}".format(len(test_ds)))


class BertLSTM(nn.Layer):
    def __init__(self, hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.5):
        super(BertLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Bert model
        self.bert = ppnlp.transformers.ElectraModel.from_pretrained('chinese-electra-small')
        for param in self.bert.parameters():
            param.stop_gradient = False

        # LSTM layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_dim, num_layers=n_layers,
                            direction="bidirectional" if bidirectional else "forward")

        # Dropout layer
        self.dropout = nn.Dropout(p=drop_prob)

        # Fully connected layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_size)

    def forward(self, input_ids, segment_ids):
        # Generate BERT embeddings
        sequence_output = self.bert(input_ids, segment_ids)
        # print(f"Shape of BERT output (sequence_output): {sequence_output.shape}")

        # LSTM layer
        lstm_out, (hidden_last, cn_last) = self.lstm(sequence_output)
        # print(f"Shape of LSTM output (lstm_out): {lstm_out.shape}")
        # print(f"Shape of hidden state (hidden_last): {hidden_last.shape}")
        # print(f"Shape of cell state (cn_last): {cn_last.shape}")
        # Dropout and fully connected layer
        lstm_out = self.dropout(lstm_out)
        if self.bidirectional:
            hidden_last_L = hidden_last[-2]
            hidden_last_R = hidden_last[-1]
            hidden_last_out = paddle.concat([hidden_last_L, hidden_last_R], axis=-1)
        else:
            hidden_last_out = hidden_last[-1]

        out = self.fc(hidden_last_out)

        return out

    def init_hidden(self, batch_size):
        number = 2 if self.bidirectional else 1
        weight = next(self.parameters())
        hidden = (paddle.zeros([self.n_layers * number, batch_size, self.hidden_dim], dtype=weight.dtype),
                  paddle.zeros([self.n_layers * number, batch_size, self.hidden_dim], dtype=weight.dtype))
        return hidden


# Data loading and training process
if __name__ == "__main__":
    tokenizer = ppnlp.transformers.ElectraTokenizer.from_pretrained('chinese-electra-small')
    label_list = ['0', '1']  # Example label list for binary classification


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


    # Data transformation function
    trans_fn = partial(convert_example, tokenizer=tokenizer, label_list=label_list, max_seq_length=256, is_test=False)

    # Batchify function
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment_ids
        Stack(dtype="int64")  # labels
    ): [data for data in fn(samples)]


    # Create dataloader
    def create_dataloader_bert(dataset, trans_fn=None, mode='train', batch_size=1, use_gpu=True, pad_token_id=0,
                               batchify_fn=None):
        if trans_fn:
            dataset = dataset.map(trans_fn, lazy=True)

        if mode == 'train' and use_gpu:
            sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=True)
        else:
            shuffle = True if mode == 'train' else False  # If not training set, do not shuffle
            sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, return_list=True, collate_fn=batchify_fn)
        return dataloader


    # Load dataset
    bert_split_data_to_train()
    train_loader = create_dataloader_bert(train_ds, mode='train', batch_size=64, batchify_fn=batchify_fn,
                                          trans_fn=trans_fn, use_gpu=True)
    dev_loader = create_dataloader_bert(dev_ds, mode='dev', batch_size=64, batchify_fn=batchify_fn, trans_fn=trans_fn,
                                        use_gpu=True)
    test_loader = create_dataloader_bert(test_ds, mode='test', batch_size=64, batchify_fn=batchify_fn,
                                         trans_fn=trans_fn, use_gpu=True)

    model = BertLSTM(hidden_dim=384, output_size=2, n_layers=2, bidirectional=True, drop_prob=0.5)
    model = paddle.DataParallel(model)

    # Set training hyperparameters
    learning_rate = 1e-6
    epochs = 3
    warmup_proption = 0.1
    weight_decay = 0.01

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(warmup_proption * num_training_steps)


    def get_lr_factor(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return max(0.0,
                       float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


    # Learning rate scheduler
    lr_scheduler = paddle.optimizer.lr.LambdaDecay(learning_rate,
                                                   lr_lambda=lambda current_step: get_lr_factor(current_step))

    # Optimizer
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    # Loss function and metric
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()


    # Evaluation function
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
            preds = paddle.argmax(probs, axis=1).numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            correct = metric.compute(probs, labels)
            metric.update(correct)
            accu = metric.accumulate()

        cm = confusion_matrix(all_labels, all_preds).tolist()
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


    # Training loop
    global_step = 0
    for epoch in range(1, epochs + 1):
        print("epoch:", epoch)
        for step, batch in enumerate(train_loader, start=1):
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
            if global_step % 50 == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (
                    global_step, epoch, step, loss, acc))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()
        eval_loss, eval_acc = evaluate(model, criterion, metric, dev_loader)
        paddle.save(model.state_dict(), f'./saved_models/electra_bilstm_model_epoch_{epoch}.pdparams')
