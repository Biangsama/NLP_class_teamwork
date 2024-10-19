import jieba
jieba.setLogLevel(0)
import pickle
import re
import warnings
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')



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
    estimator.fit(emails, labels)

    y_preds = estimator.predict(emails)
    print('训练集准确率:', accuracy_score(labels, y_preds))

    # 存储特征提取器和模型
    pickle.dump(extractor, open('D:/NLP_email/Bayes/model/extractor.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(estimator, open('D:/NLP_email/Bayes/model/estimator.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def evaluate():

    # 加载特征提取器
    extractor = pickle.load(open('D:/NLP_email/Bayes/model/extractor.pkl', 'rb'))
    # 加载算法模型
    estimator = pickle.load(open('D:/NLP_email/Bayes/model/estimator.pkl', 'rb'))
    # 加载测试集数据
    test_data = pickle.load(open('D:/NLP_email/temp/清洗训练集.pkl', 'rb'))
    emails = test_data['email']
    labels = test_data['label']
    # 提取特征
    emails = extractor.transform(emails)
    # 模型预测
    y_preds = estimator.predict(emails)
    print('验证集准确率:', accuracy_score(labels, y_preds))


class RecognizerMail:
    def __init__(self):
        # 加载特征提取器
        self.extractor = pickle.load(open('D:/NLP_email/Bayes/model/extractor.pkl', 'rb'))
        # 加载算法模型
        self.estimator = pickle.load(open('D:/NLP_email/Bayes/model/estimator.pkl', 'rb'))

    def clean_mail(self, mail):
        # 保留中文
        mail = re.sub(r'[^\u4e00-\u9fa5]', '', mail)
        # 内容分词
        mail = jieba.lcut(mail)
        return ' '.join(mail)

    def predict(self, mails):
        clean_mails = []
        # 邮件清洗
        for mail in mails:
            mail = self.clean_mail(mail)
            clean_mails.append(mail)
        # 特征提取
        input_mails = self.extractor.transform(clean_mails)
        # 模型预测
        labels = self.estimator.predict(input_mails)
        labels = ['垃圾邮件' if label == 'spam' else '正常邮件' for label in labels]
        print(labels)

        return labels


if __name__ == '__main__':
    # train()
    #evaluate()
    email1 = '''
        Received: from coozo.com ([219.133.254.230])
        by spam-gw.ccert.edu.cn (MIMEDefang) with ESMTP id j8L2Zoqi028766
        for <li@ccert.edu.cn>; Fri, 23 Sep 2005 13:01:45 +0800 (CST)
    Message-ID: <200509211035.j8L2Zoqi028766@spam-gw.ccert.edu.cn>
    From: "you" <you@coozo.com>
    Subject: =?gb2312?B?us/X9w==?=
    To: li@ccert.edu.cn
    Content-Type: text/plain;charset="GB2312"
    Content-Transfer-Encoding: 8bit
    Date: Sun, 23 Oct 2005 23:44:32 +0800
    X-Priority: 3
    X-Mailer: Microsoft Outlook Express 6.00.2800.1106
     您好！ 
           我公司有多余的发票可以向外代开！（国税、地税、运输、广告、海关缴款书）。 
        如果贵公司（厂）有需要请来电洽谈、咨询！ 
                   联系电话: 013510251389  陈先生
                                                               谢谢
    顺祝商祺!
        '''
    email2 = '''
        Received: from web15010.mail.cnb.yahoo.com (web15010.mail.cnb.yahoo.com [202.165.103.67])
        by spam-gw.ccert.edu.cn (MIMEDefang) with ESMTP id j8R8H2V8018468
        for <hu@ccert.edu.cn>; Thu, 29 Sep 2005 19:39:41 +0800 (CST)
    Received: (qmail 54688 invoked by uid 60001); Thu, 29 Sep 2005 11:50:48 -0000
    DomainKey-Signature: a=rsa-sha1; q=dns; c=nofws;
      s=s1024; d=yahoo.com.cn;
      h=Message-ID:Received:Date:From:Subject:To:MIME-Version:Content-Type:Content-Transfer-Encoding;
      b=bSU/zJOkkJfDLFBbWnWnTUKDZWedZej7CHwk+68TMJOxc5bWNOV3oFm+Sdj7+BguqbdY8hBnj9by0vLAREwvNsRCI/vWqZokpQhqNS620fenBohJKxF1JDhRipTl6dha0/sPi1Z9L+cjbm98QQkoNFkiZSBiuBy63tmjYznR3JE=  ;
    Message-ID: <20050927082809.54686.qmail@web15010.mail.cnb.yahoo.com>
    Received: from [61.150.43.113] by web15010.mail.cnb.yahoo.com via HTTP; Thu, 29 Sep 2005 19:50:48 CST
    Date: Thu, 29 Sep 2005 19:50:48 +0800 (CST)
    From: liang ming <yang@yahoo.com.cn>
    Subject: =?gb2312?B?UmU6ILOzvNzKscTQxfPT0cDPt62z9tLUx7C1xMrCx+k=?=
    To: hu@ccert.edu.cn
    MIME-Version: 1.0
    Content-Type: multipart/alternative; boundary="0-1710224003-1127809689=:53686"
    Content-Transfer-Encoding: 8bit
    我怎么觉得是你在翻..
         标  题: 吵架时男朋友老翻出以前的事情
         我觉得吵完了和好了就过去了，他却总是在下一次吵架的时候提起。是不是心胸不够宽
         阔？老这样下去伤心死了。经常是吵完了我哭他不理我，后来太晚了他就搂着我拍拍我然
         后天亮了我们都要去上班。昨天他说他想要的太多了，得到的太少了。我说我从来不觉得
         我付出的少，他就质问我付出了什么。我为了他离开了以前的男朋友，办好了去日本的签
         证而没去，离开了大连在这里辛苦的生活。远离了一些朋友，工资没怎么涨，每天忍受着
         一个半小时的公交车，饭费房租都是2倍而房子却不是精装修也没有家电，忍受着电梯和
         楼下汽车的噪音。听他那么说真伤心，觉得自己的爱在消减，好担心会不爱了。
         --
        '''
    email3 = '你好老师，我今天可以请假一天吗'
    recognizer = RecognizerMail()
    recognizer.predict([email1, email2, email3])