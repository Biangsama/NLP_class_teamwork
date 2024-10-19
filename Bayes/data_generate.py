import time
from sklearn.model_selection import train_test_split
from collections import Counter
import codecs
from tqdm import tqdm
import pickle
import multiprocessing
from joblib import Parallel
from joblib import delayed
import os
import re
from datasets import load_dataset
import pandas as pd
import os
import codecs
import re
import zhconv
import jieba
import jieba.posseg as psg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import time
from tqdm import tqdm


# 将当前目录设置为工作目录
#os.chdir(os.path.dirname(os.path.abspath(.)))

# 结巴不输出日志
jieba.setLogLevel(jieba.logging.INFO)

def read_mail_data():

    # 读取目录
    fnames, labels = [], []
    for line in open('D:/NLP_email/trec06c/full\index'):
        label, path = line.strip().split()
        # 读取的路径: ../data/215/104
        # 转换为路径: data/data/215/104
        path = path.replace('..', 'D:/NLP_email/trec06c')
        fnames.append(path)
        labels.append(label)

    # 读取文件
    emails = [open(fname, encoding='gbk', errors='ignore').read() for fname in fnames]
    # 数据分布
    print('数据集分布:', dict(Counter(labels)))
    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(emails, labels, test_size=0.1, random_state=22, stratify=labels)
    print('训练集分布:', dict(Counter(y_train)))
    print('测试集分布:', dict(Counter(y_test)))

    pickle.dump({'email': x_train, 'label': y_train}, open('D:/NLP_email/temp/原始训练集.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump({'email': x_test,  'label': y_test}, open('D:/NLP_email/temp/原始测试集.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

read_mail_data()