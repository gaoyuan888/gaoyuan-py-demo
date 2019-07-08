import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import operator
from gensim.models import KeyedVectors
import codecs
import json
import jieba
import pandas as pd

diag_list = pd.read_csv('data/23728146.csv')
diag_list = diag_list.dropna(subset=["reception_doctor_id", "disease_desc"])
disease_desc_list = diag_list["disease_desc"]
reception_doctorid_list = diag_list["reception_doctor_id"]

# 疾病库
disease_set = set(("关节炎",))
# 医生：疾病字典
doc_disease_dict = {}

# 按类别：疾病 字典
class_disease_dict = {}

# write_ = codecs.open("sentence_similar_array.txt", 'w', encoding="utf8")
for idx in range(disease_desc_list.__len__()):
    disease_desc = disease_desc_list[idx]
    disease_desc = re.sub("\t", "", disease_desc)
    if disease_desc.startswith("线下确诊疾病为"):
        d_l = disease_desc.split("；")[0].split("：")[1].split("、")
        if doc_disease_dict.__contains__(reception_doctorid_list[idx]):
            doc_disease_dict[reception_doctorid_list[idx]] += set(doc_disease_dict[reception_doctorid_list[idx]] + d_l)
        for d in d_l:
            disease_set.add(d)
        print(d_l)

print(disease_set)
print(diag_list.shape)

sss = jieba.cut("你好我好大家好", cut_all=True)
for ss in sss:
    print(ss)

b = [1, 0.8, 1]
sssssss = sorted(np.array(b), reverse=True)
list()

fdsa = sorted(b)
ssssss = str(b)
write_ = codecs.open("feature_words_write.txt", 'w', encoding="utf8")
with open("./goodat_cluster.txt", 'r', encoding='utf-8') as load_f:
    strF = load_f.read()
    if len(strF) > 0:
        goodat_list = json.loads(strF)
    print(goodat_list)
    for good_at_arr in goodat_list:
        line_list = []
        for good_at in good_at_arr:
            line_list += corpus_list[good_at.line_id].split(" ")
        write_.writelines(line_list + "\n")
write_.close()

b = [1, 0.8, 1]
c = [1, 1, 1]
cc = b + c
# array_ = np.loadtxt("./data.txt")
# list_ = list(array_)
with open("./goodat_cluster.txt", 'r', encoding='utf-8') as load_f:
    strF = load_f.read()
    if len(strF) > 0:
        datas = json.loads(strF)
    else:
        datas = {}
    print(datas)

# 打开一个json文件
data = open("./goodat_cluster.txt", encoding='utf-8')
# 转换为python对象
strJson = json.load(data)
print(strJson[0][0]['line_id'])
for item in strJson:
    print(item)

b = [1, 0.8, 1]
c = [1, 1, 1]

dd = []
dd.append(a)
dd.append(b)
A_sparse_onehot = sparse.csr_matrix(dd)
sss = cosine_similarity(dd)

cn_model = KeyedVectors.load_word2vec_format('data/sgns.zhihu.bigram', binary=False)
ss = cn_model["便秘"]
dd = cn_model["肛肠"]
ee = cn_model["衣服"]
dee = cn_model["咳嗽"]
ddsss = np.array([dd, ee, dee], dtype=np.float)
dsadsa = cn_model.cosine_similarities(ss, ddsss).tolist()

dfda = cn_model.similarity("便秘", "肛肠")

sss = np.zeros(shape=[12000, ])
sss[0] = 1
print(sss)

print(int(5.3))

disease_desc = "fdsafdafda"
disease_desc = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+|[a-zA-Z0-9_]+", "", disease_desc)

dictss = {}
dictss[1] = 3
dictss[2] = 4
class_total_dict = sorted(dictss.items(), key=operator.itemgetter(1), reverse=True)
print(dictss.__contains__(3))


class Person:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    def __str__(self):
        return "name: {}, age: {}".format(self.__name, self.__age)


p = Person('范冰冰', 37)
print(p.__str__())
sss = dict(p.__str__())
print(sss)
dictss = "{" + p.__str__() + "}"
print(dictss[1])
print(Person('范冰冰', 37))

people = [Person('范冰冰', 37), Person('柳岩', 36), Person('王菲', 47)]

a = ["s", "d", "s"]
b = ["b", "b", "d"]
print(a + b)

print(sss)
print(np.sum([a, b, c], axis=0))

tag_list = ['青年 吃货 唱歌',
            '叛逆 游戏 叛逆',
            '少年 吃货 足球']

vectorizer = CountVectorizer()  # 将文本中的词语转换为词频矩阵
X = vectorizer.fit_transform(tag_list)  # 计算词语出现的次数 3x7 sparse matrix
"""
word_dict = vectorizer.vocabulary_
{'唱歌': 2, '吃货': 1, '青年': 6, '足球': 5, '叛逆': 0, '少年': 3, '游戏': 4}
"""
print(X.toarray())
'''
array([[0, 1, 1, 0, 0, 0, 1],
       [1, 0, 0, 1, 1, 0, 0],
       [0, 1, 0, 1, 0, 1, 0]])
'''
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)  # 将词频矩阵X统计成TF-IDF值
print(tfidf.toarray())
'''
[[0.        0.4736296        0.62276601        0.        0.        0.       0.62276601]
 [0.62276601        0.        0.        0.4736296        0.62276601        0.        0.]
 [0.        0.51785612        0.        0.51785612        0.        0.68091856        0.]]
'''

a = [1] * 1300
b = [0] * 1300
a = re.sub("['[\]'\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", str(a))
b = re.sub("['[\]'\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", str(b))

d = list(str(int(a) | int(b)))
print(d)
c = 101010
b = 101011
print(c | b)
ss = '.'.join(a)
dd = '.'.join(b)
print(ss & dd)

sss = []
tup1 = (1,)

a = [[1, 0, 0], [1, 0, 1]]
a = [1, 1, 1]
b = [2, 2, 2]

print(np.sum([a, b], axis=0))
cdee = [0, 0, 0]
count = 0
for arr in abc:
    for index in range(arr.__len__()):
        cdee[index] = 1 if arr[index] & cde[index] == 1 else cdee[index]

print(cdee)

sss.append(tup1)

tup2 = tup1 + (2,)
print(tup2)
sss[0] = tup2
print(sss)

tup1 = (12, 34.56)
tup2 = ('abc', 'xyz')
tup3 = tup1 + tup2  # python运行元组进行连接组合
print(tup3)  # 输出:(12,34.56,'abc','xyz')

info_tuple_01 = ("zhangsan", 18, 1.75)
tuple_list = []
tuple_list

import codecs

# 使用迭代遍历元组
for my_info in info_tuple_01:
    # 使用格式字符串彬姐 my_info 这个变量不方便
    # 因为元组中通常保存的数据类型是不同的
    print(my_info)

original_file = "diag.csv"
# newfile=original_file[0:original_file.rfind(.)]+'_copy.csv'
f = open(original_file, 'rb+')
content = f.read()  # 读取文件内容，content为bytes类型，而非string类型
source_encoding = 'utf-8'
#####确定encoding类型
try:
    content.decode('utf-8').encode('utf-8')
    source_encoding = 'utf-8'
except:
    try:
        content.decode('gbk').encode('utf-8')
        source_encoding = 'gbk'
    except:
        try:
            content.decode('gb2312').encode('utf-8')
            source_encoding = 'gb2312'
        except:
            try:
                content.decode('gb18030').encode('utf-8')
                source_encoding = 'gb18030'
            except:
                try:
                    content.decode('big5').encode('utf-8')
                    source_encoding = 'gb18030'
                except:
                    content.decode('cp936').encode('utf-8')
                    source_encoding = 'cp936'
f.close()

#####按照确定的encoding读取文件内容，并另存为utf-8编码：
block_size = 4096
with codecs.open(original_file, 'r', source_encoding) as f:
    with codecs.open('diag.csv', 'w', 'utf-8') as f2:
        while True:
            content = f.read(block_size)
            if not content:
                break
            f2.write(content)
