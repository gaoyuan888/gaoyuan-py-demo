# coding=gbk
# %matplotlib inline
# gensim用来加载预训练word vector
import warnings
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import jieba  # 结巴分词
import numpy as np
import operator
warnings.filterwarnings("ignore")

# 一、词向量模型
cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', binary=False)

# step 1 读取停用词
stop_words = []
with open('stop_words.txt', encoding='utf-8') as f:
    line = f.readline()
    while line:
        stop_words.append(line[:-1])
        line = f.readline()
stop_words = set(stop_words)
print('停用词读取完毕，共{n}个单词'.format(n=len(stop_words)))

# step2 读取文本，预处理，分词，得到词典
raw_word_list = []
sentence_list = []
# 构造一个字典存取数据
cluster_dict = {}
dict_id = 0
line_id=0
with open('med.zh.txt', encoding='utf-8') as f:
    line = f.readline()
    print("line_index:{}",line_id)
    line_id=line_id+1
    while line:
        while '\n' in line:
            line = line.replace('\n', '')
        while ' ' in line:
            line = line.replace(' ', '')
        if len(line) > 0:  # 如果句子非空
            raw_words = set(jieba.cut(line, cut_all=False))
            dealed_words = []
            for word in raw_words:
                if word not in stop_words and word not in ['qingkan520', 'www', 'com', 'http']:
                    raw_word_list.append(word)
                    dealed_words.append(word)
            sentence_list.append(dealed_words)
            # 对比词语相似度。组装字段并分类
            if cluster_dict.__len__() == 0:
                list_tmp = []
                list_tmp.append(dealed_words)
                cluster_dict[dict_id] = list_tmp
                dict_id = dict_id + 1
            else:
                # 遍历字典，对比词语的相似度，
                # 存在某个字典key下的list，其中一定比例的list所有的词语，有大于某个比例的词语相似读大于0.6，则认为词条是相似的
                add_flag = False
                # 保存字典相似百分比
                same_num_dict = {}
                similar_percent_dict = {}
                decision_percent_dict = {}
                for key, value in cluster_dict.items():
                    print(key, ' value : ', value)
                    # 定义一个矩阵用来记录相似度
                    similar_list = []
                    sum_value = 0
                    similar_num_total = 0
                    # 此时，value 是list of list 格式
                    for value_sentense in value:
                        similar_value_set = []
                        similar_dealed_set = []
                        for value_word in value_sentense:
                            sum_value = sum_value+1
                            for dealed_word in dealed_words:
                                try:
                                    similar_of_words = cn_model.similarity(value_word, dealed_word)
                                    similar_list.append(similar_of_words)
                                    if similar_of_words > 0.6:
                                        similar_value_set = set(similar_value_set.append(value_word))
                                        similar_dealed_set = set(similar_dealed_set.append(dealed_word))
                                except KeyError as e:
                                    pass
                        similar_num_total = similar_value_set.__len__()+similar_dealed_set.__len__()

                    # 计算相似度大于0.6的个数，如果大于一半以上，则认为两个句子是属于一类的，则添加到当前的key中，否则的话添加新的字典中
                    similar_size = similar_list.__len__()
                    similar_num = 0
                    same_num = 0
                    similar_sum_total = value.__len__()*dealed_words.__len__()+sum_value
                    for sim in similar_list:
                        if sim >= 0.999:
                            same_num = same_num+1
                        if sim > 0.6:
                            similar_num = similar_num+1
                    same_num_dict[key] = same_num
                    similar_percent_dict[key] = similar_num/similar_size
                    decision_percent_dict[key] = same_num * similar_num/similar_size
                    print("same_size:{},sum_value:{}", same_num, sum_value)

                dict_list = sorted(similar_percent_dict.items(), key=operator.itemgetter(1), reverse=True)
                print(similar_size, dict_list)
                same_list = sorted(same_num_dict.items(), key=operator.itemgetter(1), reverse=True)
                print(sum_value, same_list)
                print(similar_size, sorted(decision_percent_dict.items(), key=operator.itemgetter(1), reverse=True))

                if dict_list[0][1] <= 0.2 or same_list[0][1] < 3:
                    # 说明没有相同的，则新开启一个字典记录
                    list_tmp = []
                    list_tmp.append(dealed_words)
                    cluster_dict[dict_id] = list_tmp
                    dict_id = dict_id + 1
                else:
                    cluster_dict[same_list[0][0]].append(dealed_words)

        line = f.readline()

print(cluster_dict.__len__())

# 1.词向量维度，一个词向用300个维度向量表示，embedding_dim=300
embedding_dim = cn_model['山东大学'].shape[0]
print('词向量长度{}', format(embedding_dim))
array = cn_model['山东大学']
# 打印词向量
print("array", array)

# 2.相似度，余弦相似度
similar = cn_model.similarity('婴幼儿', '婴幼儿')
print("similar", similar)

# 余弦相似度计算方法    点积（矩阵/范数，矩阵/范数）  dot(["橘子"]/|["橘子"],["橘子"]/|["橘子"])
similar_ = np.dot(cn_model["橘子"] / np.linalg.norm(cn_model["橘子"]), cn_model["橙子"] / np.linalg.norm(cn_model["橙子"]))
print("similar_", similar_)

# 3.寻找不是同一类的词语
test_words = "新生儿 婴儿 儿童 老人"
test_words_res = cn_model.doesnt_match(test_words.split())
print('在' + test_words + "中，不是同一类的词为：%s" % test_words_res)

# 4.找出相近的词语，余弦相似度
similar_1 = cn_model.most_similar(positive=['儿科'], topn=10)
print("similar_1", similar_1)

# 5.求一个词语在词向量中的索引
index1 = cn_model.vocab["新生儿"].index
print("index1", index1);

# 6.根据索引求索引的词语  215 对应的此为老师
word = cn_model.index2word[215];
print("word", word);
