from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import joblib

BYS = MultinomialNB()

def loadfile(filepath='training.txt'):#读取数据，分割数据
    with open(filepath, 'r',encoding='utf-8') as f:
        aim = f.readline()#读每一行数据
        target = []
        features = []
        while aim:
            first, second = aim.split('\t')#根据TAB分割，first=好评/差评，secennd=评价内容
            target.append(first)
            features.append(second)
            aim = f.readline()
    return (target, features)


tags, data = loadfile()
#max_df = 0.50表示“忽略出现在50％以上文档中的术语”.
#max_df = 25表示“忽略超过25个文档中出现的术语”，min不等号反向
#不忽略，全部读取
#初始化读取方法
vectoerizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')
# 1.CountVectorizer
# fit(raw_documents) :根据CountVectorizer参数规则进行操作，比如滤除停用词等，拟合原始数据，生成文档中有价值的词汇表；
#
# transform(raw_documents):使用符合fit的词汇表或提供给构造函数的词汇表，从原始文本文档中提取词频，转换成词频矩阵。
#
# fit_transform(raw_documents, y=None):学习词汇词典并返回术语 - 文档矩阵(稀疏矩阵)。
#
# 2.TfidfTransformer，TF-IDF(Term frequency * Inverse Doc Frequency)词权重
#在较低的文本语料库中，一些词非常常见（例如，英文中的“the”，“a”，“is”），因此很少带有文档实际内容的有用信息。如果我们将单纯的计数数据直接喂给分类器，那些频繁出现的词会掩盖那些很少出现但是更有意义的词的频率。
# 为了重新计算特征的计数权重，以便转化为适合分类器使用的浮点值，通常都会进行tf-idf转换。词重要性度量一般使用文本挖掘的启发式方法：TF-IDF。IDF，逆向文件频率（inverse document frequency）是一个词语普遍重要性的度量（不同词重要性的度量）。
#Tf表示术语频率，而 tf-idf 表示术语频率乘以逆向文件频率:
#
#fit(raw_documents, y=None)：根据训练集生成词典和逆文档词频 由fit方法计算的每个特征的权重存储在model的idf_属性中。
#
#transform(raw_documents, copy=True)：使用fit（或fit_transform）学习的词汇和文档频率（df），将文档转换为文档 - 词矩阵。返回稀疏矩阵，[n_samples, n_features]，即，Tf-idf加权文档矩阵（Tf-idf-weighted document-term matrix）。
#
#
#
#将文本中的词语转换为词频矩阵，矩阵元素matrix[i][j] 表示j词在第i个文本下的词频
matrix = vectoerizer.fit_transform(data)

#统计每个词语的tf-idf权值
train_tfidf = TfidfTransformer(use_idf=False).fit_transform(matrix)
BYS.fit(train_tfidf, tags)#进行贝叶斯分类
joblib.dump(BYS,  "2017081267_moxing.txt") #导出模型

_, testset = loadfile('test.txt')#加载测试数据文件
vector = vectoerizer.transform(testset)
fidf= TfidfTransformer(use_idf=False).fit_transform(vector)


lr = joblib.load("2017081267_moxing.txt") #导入模型
results = lr.predict(fidf)#进行预测
with open('2017081267_预测结果.txt', 'w',encoding='utf-8') as f:#输出结果
    for num, result in enumerate(results,  1):
       f.write(str(num) +" " + result+ '\n')
