# coding:utf-8
# 数据集
from sklearn import datasets

news = datasets.fetch_20newsgroups(subset='all')
# 查看一下第一条新闻的内容和分组
print(news.keys())
# ['DESCR', 'data', 'target', 'target_names', 'filenames']

# 划分训练集和测试集，分为80%训练集，20%测试集
split_rate = 0.8
split_size = int(len(news.data) * split_rate)
X_train = news.data[:split_size]
y_train = news.target[:split_size]
X_test = news.data[split_size:]
y_test = news.target[split_size:]

# 特征提取
# --为了使机器学习算法应用在文本内容上，首先应该把文本内容装换为数字特征。这里使用词袋模型(Bags of words)
# 词袋模型
# --在信息检索中，Bag of words model假定对于一个文本，忽略其词序和语法，句法，
# --将其仅仅看做是一个词集合，或者说是词的一个组合，文本中每个词的出现都是独立的，
# --不依赖于其他词是否出现，或者说当这篇文章的作者在任意一个位置选择一个词汇都不受前面句子的影响而独立选择的
# CountVectorizer、TfidfTransformer 这部分内容参考博客 https://blog.csdn.net/Eastmount/article/details/50323063
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Tokenizing text
count_vect = CountVectorizer()
# 每个次在训练集中出现的次数
X_train_counts = count_vect.fit_transform(X_train)
# Tf
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
# Tf_idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# 稀疏性`
# --大多数文档通常只会使用语料库中所有词的一个子集，因而产生的矩阵将有许多特征值是0（通常99%以上都是0）。
# --例如，一组10,000个短文本（比如email）会使用100,000的词汇总量，而每个文档会使用100到1,000个唯一的词。
# --为了能够在内存中存储这个矩阵，同时也提供矩阵/向量代数运算的速度，通常会使用稀疏表征
# --例如在scipy.sparse、包中提供的表征。


# 训练模型
# --上面使用文本中词的出现次数作为数值特征，
# --可以使用多项分布估计这个特征，
# --使用sklearn.naive_bayes模块的MultinomialNB类来训练模型。
# nbc means naive bayes classifier
from sklearn.naive_bayes import MultinomialNB

# create classifier
clf = MultinomialNB().fit(X_train_tfidf, y_train)
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# using classifier to predict
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, news.target_names[category]))

# 使用Pipline这个类构建复合分类器
# Scikit-learn为了使向量化 => 转换 => 分类这个过程更容易，提供了Pipeline类来构建复合分类器，例如：
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, \
    CountVectorizer

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ])

# 创建新的训练模型
nbc_1 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])
nbc_2 = Pipeline([
    ('vect', HashingVectorizer(non_negative=True)),
    ('clf', MultinomialNB()),
])
nbc_3 = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])
# classifier
nbcs = [nbc_1, nbc_2, nbc_3]

# 下面是一个交叉验证函数：
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from scipy.stats import sem
import numpy as np

# cross validation function
def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator of k folds
    cv = KFold(K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))


# 将训练集分为10份，输出验证分数：
for nbc in nbcs:
    evaluate_cross_validation(nbc, X_train, y_train, 10)
# 结果为：
# CountVectorizer Mean score: 0.849 (+/-0.002)
# HashingVectorizer Mean score: 0.765 (+/-0.006)
# TfidfVectorizer Mean score: 0.848 (+/-0.004)
# 可以看出：CountVectorizer和TfidfVectorizer特征提取的方法要比HashingVectorizer效果好。


# 优化模型

## 优化单词提取
# --在使用TfidfVectorizer特征提取时候，使用正则表达式，
# --默认的正则表达式是u'(?u)\b\w\w+\b'，
# --使用新的正则表达式ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b"
nbc_4 = Pipeline([
    ('vect', TfidfVectorizer(
        token_pattern="\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b", )
     ),
    ('clf', MultinomialNB()),
])
evaluate_cross_validation(nbc_4, X_train, y_train, 10)
# 分数是：Mean score: 0.861 (+/-0.004) ，结果好了一点

## 排除停止词
# TfidfVectorizer的一个参数stop_words，
# 这个参数指定的词将被省略不计入到标记词的列表中，
# 这里使用鼎鼎有名的NLTK语料库。
import nltk

# nltk.download()
stopwords = nltk.corpus.stopwords.words('english')
nbc_5 = Pipeline([
    ('vect', TfidfVectorizer(
        stop_words=stopwords,
        token_pattern="\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB()),
])
evaluate_cross_validation(nbc_5, X_train, y_train, 10)
# 分数是：Mean score: 0.879 (+/-0.003)，结果又提高了

## 调整贝叶斯分类器的alpha参数
# MultinomialNB有一个alpha参数，该参数是一个平滑参数，默认是1.0，我们将其设为0.01
nbc_6 = Pipeline([
    ('vect', TfidfVectorizer(
        stop_words=stopwords,
        token_pattern="\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB(alpha=0.01)),
])
evaluate_cross_validation(nbc_6, X_train, y_train, 10)
# 分数为：Mean score: 0.917 (+/-0.002)，
# 哎呦，好像不错哦！不过问题来了，调整参数优化不能靠蒙，
# 如何寻找最好的参数，使得交叉验证的分数最高呢？


## 使用Grid Search优化参数
# 使用GridSearch寻找vectorizer词频统计, tfidftransformer特征变换和MultinomialNB classifier的最优参数
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
]);
parameters = {
    'vect__max_df': (0.5, 0.75),
    'vect__max_features': (None, 5000, 10000),
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001),
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=1)

from time import time

t0 = time()
grid_search.fit(X_train, y_train)
print ("done in %0.3fs" % (time() - t0))
print ("Best score: %0.3f" % grid_search.best_score_)

# #输出最优参数

best_parameters = dict()
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ("\t%s: %r" % (param_name, best_parameters[param_name]))
pipeline.set_params(clf__alpha=1e-05,
                    tfidf__use_idf=True,
                    vect__max_df=0.5,
                    vect__max_features=None)
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

# 经过漫长的等待，终于找出了最优参数：
# done in 1578.965s
# Best score: 0.902
#
# clf__alpha: 0.01
# tfidf__use_idf: True
# vect__max_df: 0.5
# vect__max_features: None
# 在测试集上的准确率为：0.915，分类效果还是不错的

from sklearn import metrics

print (np.mean(pred == y_test))

# print X_test[0], y_test[0]
for i in range(20):
    print (str(i) + ": " + news.target_names[i])
predicted = pipeline.fit(X_train, y_train).predict(X_test)
print(np.mean(predicted == y_test))
print (metrics.classification_report(y_test, predicted))
