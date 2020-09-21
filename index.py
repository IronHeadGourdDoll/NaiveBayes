from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer#导入特征提取的包
from sklearn.naive_bayes import MultinomialNB#导入贝叶斯分类的包
import numpy as np
import os
BYS = MultinomialNB()

def loadfile(filepath='training.txt'):#读取数据，分割数据
    with open(filepath, 'r',encoding='utf-8') as f:
        aim = f.readline()
        target = []
        features = []
        while aim:
            first, second = aim.split('\t')
            target.append(first)
            features.append(second)
            aim = f.readline()
    return (target, features)

def getresult(filepath='test.txt'):#获得特征结果
    with open(filepath, 'r', encoding='utf-8') as f:
        features = []
        aim = f.readline()
        while aim:
            features.append(aim)
            aim = f.readline()
    return features

tags, data = loadfile()
vectoerizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')#将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i个文本下的词频
matrix = vectoerizer.fit_transform(data)#对数据进行某种统一处理
train_tfidf = TfidfTransformer(use_idf=False).fit_transform(matrix)#统计每个词语的tf-idf权值
BYS.fit(train_tfidf, tags)#进行贝叶斯分类

_, testset = loadfile('test.txt')
vector = vectoerizer.transform(testset)
fidf= TfidfTransformer(use_idf=False).fit_transform(vector)

results = BYS.predict(fidf)#进行预测
for num, result in enumerate(results):
    print (num, "  ", result)