# coding:utf-8
__author__ = "liuxuejiang"
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

reload(sys)
sys.setdefaultencoding('utf8')




if __name__ == "__main__":
    # corpus = [u"我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
    #           u"他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
    #           u"小明 硕士 毕业 与 中国 科学院",  # 第三类文本的切词结果
    #           u"我 爱 北京 天安门"]  # 第四类文本的切词结果
    #
    # vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    # vec_count = vectorizer.fit_transform(corpus)
    # word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    # print vec_count.toarray()
    # print word
    # # tf
    # tf = TfidfTransformer(use_idf=False).fit(vec_count)
    # X_train_tf = tf.transform(vec_count)
    #
    # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    # tfidf = transformer.fit_transform(vec_count)  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    #
    # weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    # for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    #     print u"-------这里输出第", i, u"类文本的词语tf-idf权重------"
    #     for j in range(len(word)):
    #         print word[j], weight[i][j]


    print "======================================================================================="
    print "======================================================================================="
    # 语料
    corpus = [
        'This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?',
    ]
    # 将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer()
    # 计算个词语出现的次数
    vec_count = vectorizer.fit_transform(corpus)
    # 获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()
    print word
    # 查看词频结果
    print vec_count.toarray()

    # tf
    tf_transformer = TfidfTransformer(use_idf=False).fit(vec_count)
    X_train_tf = tf_transformer.transform(vec_count)
    print X_train_tf
    # [
    # [0.         0.4472136  0.4472136  0.4472136  0.         0.            0.4472136  0.         0.4472136]
    # [0.         0.35355339 0.         0.35355339 0.         0.70710678    0.35355339 0.         0.35355339]
    # [0.5         0.         0.         0.         0.5        0.            0.5        0.5        0.        ]
    # [0.         0.4472136  0.4472136  0.4472136  0.         0.             0.4472136  0.         0.4472136]
    # ]


    print "tf-idf============================"
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(vec_count)
    print X_train_tfidf



