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

# 邮件内容清洗
def clean_email(content):

    import jieba
    jieba.setLogLevel(0)
    # 保留中文
    # content = re.sub(r'[^\u4e00-\u9fa5]', '', content)
    # 内容分词
    content = jieba.lcut(content)

    return ' '.join(content)


def clean_train_data(input_emails, input_labels, worker_count=None):

    # 数据较多，清洗需要10分钟左右
    data_count = len(input_labels)
    # 设置 cpu 数量
    worker_count = worker_count if worker_count is not None else multiprocessing.cpu_count()
    # 设置子任务数据
    task_range = list(range(0, data_count + 1, int(data_count / worker_count)))

    def task(s, e):
        emails = input_emails[s:e]
        labels = input_labels[s:e]
        result_emails = []
        result_labels = []
        progress = tqdm(range(len(labels)), desc='进程 %6d 数据区间 (%5d, %5d)' % (os.getpid(), s, e))
        filter_number = 0
        for email, label in zip(emails, labels):
            email = clean_email(email)
            progress.update(1)
            if len(email) == 0:
                filter_number += 1
                continue
            result_emails.append(email)
            result_labels.append(label)
        progress.close()

        return {'email': result_emails, 'label': result_labels, 'filter': filter_number}

    delayed_tasks = []
    for index in range(1, len(task_range)):
        s = task_range[index - 1]
        e = task_range[index]
        delayed_tasks.append(delayed(task)(s, e))

    # 多任务运行任务
    # 16核心需要81秒，6核需要140秒
    results = Parallel(n_jobs=worker_count)(delayed_tasks)
    # 合并计算结果
    clean_emails = []
    clean_labels = []
    clean_number = 0
    for result in results:
        clean_emails.extend(result['email'])
        clean_labels.extend(result['label'])
        clean_number += result['filter']

    print('训练集数量:', data_count)
    print('清理后数量:', len(clean_labels), len(clean_emails))
    print('被清理数量:', clean_number)

    # 预览数据样式
    print('预览数据:', clean_emails[0])
    print('预览标签:', clean_labels[0])

    return {'email': clean_emails, 'label': clean_labels}


def prepare_mail_data():

    train_data = pickle.load(open('D:/NLP_email/temp/原始训练集.pkl', 'rb'))
    test_data = pickle.load(open('D:/NLP_email/temp/原始测试集.pkl', 'rb'))

    train_result = clean_train_data(train_data['email'], train_data['label'])
    test_result =  clean_train_data(test_data['email'],  test_data['label'])

    # 存储清洗结果
    pickle.dump(train_result, open('D:/NLP_email/temp/清洗训练集.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_result, open('D:/NLP_email/temp/清洗测试集.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

prepare_mail_data()