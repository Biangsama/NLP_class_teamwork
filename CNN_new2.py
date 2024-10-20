import pandas as pd
import matplotlib.pyplot as plt
import jieba
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import numpy as np

# 读取停用词列表
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file if line.strip()]
    return stopwords

# 数据预处理函数
def textParse(text):
    listOfTokens = jieba.lcut(text)
    newList = [re.sub(r'\W*', '', s) for s in listOfTokens]
    filtered_text = [tok for tok in newList if len(tok) > 0]
    return filtered_text

def remove_stopwords(tokens, stopword_list):
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def normalize_corpus(corpus, stopword_list):
    normalized_corpus = []
    for text in corpus:
        filtered_text = textParse(text)
        filtered_text = remove_stopwords(filtered_text, stopword_list)
        normalized_corpus.append(filtered_text)
    return normalized_corpus

# 加载停用词表
stopword_list = load_stopwords('stop_words.txt')

# 读取CSV文件
df = pd.read_csv('mailChinese_header1.csv')

# 提取标题和标签
titles = df['DecodedSubject'].astype(str).values
labels = df['isSpam'].values

# 数据预处理
norm_titles = normalize_corpus(titles, stopword_list)

# 提取TF-IDF特征
tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2), norm='l2', smooth_idf=True, use_idf=True)
tfidf_features = tfidf_vectorizer.fit_transform(norm_titles)

# 将特征转换为适合 CNN 的形状 (样本数, 时间步, 特征数)
X = tfidf_features.toarray()
X = np.expand_dims(X, axis=1)  # 将数据转换为 (样本数, 1, 特征数)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.4, random_state=42)

# 计算类权重，处理数据不平衡问题
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# CNN模型
def create_cnn_model(input_shape):
    model = Sequential()
    # 输入层
    model.add(Input(shape=input_shape))
    # 1D 卷积层 1
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    # 最大池化层
    model.add(GlobalMaxPooling1D())
    # Dropout层防止过拟合
    model.add(Dropout(0.5))
    # 全连接层 1
    model.add(Dense(64, activation='relu'))
    # Dropout层
    model.add(Dropout(0.5))
    # 全连接层 2
    model.add(Dense(32, activation='relu'))
    # 输出层（二元分类，使用sigmoid激活函数）
    model.add(Dense(1, activation='sigmoid'))
    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 获取输入特征的形状
input_shape = (X_train.shape[1], X_train.shape[2])

# 创建CNN模型
model = create_cnn_model(input_shape)


# 模型保存路径和早停策略
checkpoint = ModelCheckpoint('best_model.weights.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

# 计算类别权重
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# 将权重转换为字典格式
class_weights = {i: weights[i] for i in range(len(weights))}

# 模型训练
# history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test),
#                     callbacks=[checkpoint, early_stopping], class_weight=class_weights)

# 评估模型
model.load_weights('best_model.weights.keras')  # 加载最优模型权重
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy:.4f}")

# # 绘制损失
# plt.figure(figsize=(12, 4))
#
# # 绘制训练和验证损失
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('CNN Loss over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# # 绘制训练和验证准确率
# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('CNN Accuracy over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# # 显示图形
# plt.tight_layout()
# plt.show()

# 预测函数
def predict_spam(title, model, vectorizer, stopword_list):
    # 数据预处理
    norm_title = normalize_corpus([title], stopword_list)
    # 提取TF-IDF特征
    tfidf_title = vectorizer.transform(norm_title).toarray()
    tfidf_title = np.expand_dims(tfidf_title, axis=1)  # 调整形状为 (1, 1, 特征数)
    # 模型预测
    prediction = model.predict(tfidf_title)
    print(f"预测概率: {prediction[0][0]}")
    return '正常邮件' if prediction[0][0] > 0.5 else '垃圾邮件'

# 假设我们已经定义了以下变量
# model: 训练好的模型
# tfidf_vectorizer: TF-IDF 向量化器
# stopword_list: 停用词列表

# # 测试标题列表
# test_titles = [
#     'Re: 帮帮忙啊',
#     '老师我今天想请一天假',
#     '免费开发票，有需求请联系055-855412',
#     '恭喜您赢得了500元购物卡!',
#     '关于下周会议的安排',
#     '快来领取您的免费礼品',
#     '再次确认您的订单信息',
#     '下周会议安排',
#     '生日聚会邀请',
#     '项目进展更新',
#     '感谢您的支持',
#     '寒假旅行计划',
#     '新员工介绍',
#     '周五的团建活动',
#     '请确认出席',
#     '季度财务报告',
#     '更新个人信息',
#     '咖啡约会邀请',
#     '志愿者活动通知',
#     '招聘信息',
#     '订票确认',
#     '年度总结会议',
#     '家庭聚会安排',
#     '课程注册提醒',
#     '感谢您的反馈',
#     '健康检查通知',
#     '推荐阅读材料'
# ]
# # 循环遍历每个标题进行预测
# for title in test_titles:
#     result = predict_spam(title, model, tfidf_vectorizer, stopword_list)
#     print(f"标题: '{title}' 被预测为: {result}")

