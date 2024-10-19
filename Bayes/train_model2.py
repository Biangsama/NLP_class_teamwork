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


def train():
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
    batch_size = 5  # 每批次的大小
    num_batches = (emails.shape[0] + batch_size - 1) // batch_size  # 计算总批次数

    for i in range(num_batches):
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


def evaluate():
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

    # 打印评估指标
    print('验证集准确率:', accuracy_score(labels, y_preds))
    print('精确率:', precision_score(labels, y_preds, average='weighted'))
    print('召回率:', recall_score(labels, y_preds, average='weighted'))
    print('F1-score:', f1_score(labels, y_preds, average='weighted'))
    print('混淆矩阵:\n', confusion_matrix(labels, y_preds))


# 调用函数
if __name__ == '__main__':
    read_mail_data()  # 读取数据并分割
    train()  # 训练模型
    evaluate()  # 评估模型
