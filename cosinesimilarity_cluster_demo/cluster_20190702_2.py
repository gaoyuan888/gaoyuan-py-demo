# gensim用来加载预训练word vector
# 待优化：某一类含有科室名称后，不再添加另外的科室名称
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import operator
import numpy as np
import pandas as pd
import codecs
import jieba
# 字符串处理
import re
# 词频统计
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import json  # 引入模块

# 定义全局变量
# 语料onehot 编码，已经填充了近义词
one_hot_array_similar = []
# 语料onehot 编码，没有组装近义词
one_hot_array = []
# 语料的list
corpus_list = []  # ['种植 牙 牙周病 治疗 修复 及口 内 治疗', '乳腺 肿瘤 及 乳房 整形 领域 的 手术 消化系统 等 常见 肿瘤 的 诊治']
# 语料的全部分词结果
corpus_tockens_list = []  # ["种植","牙"]
# 语料分组 组织一个map 进行分组,存储分类信息
goodat_cluster_list = []  # [[(0, '\t足踝外科', 0)],[(1, '\t种植科', 1), (2, '\t种植科', 1), (3, '\t种植科', 1)]]
feature_words_disk = {}
# 语料id-word字典
corpus_id2word = []
corpus_word2id = []

# 句子相似度矩阵
sentence_similar_array = []
# 每一类特征的onehot编码
feature_onehot_list = []
# 医生id和类别对应字典
docid_class_dict = {}

cluster_similar_dict = {}
cluster_similar_mult_samenum_dict = {}
cluster_same_dict = {}

regstr = "[\s+\.\!\/_,$%^*(+\"\'\t]+|[+——！，。？、~@#￥%……&*（）]+|[a-zA-Z0-9_]+"


class GoodAtCluster:
    def __init__(self, line_id, second_depart_name, cluster_idx, doctor_id):
        self.line_id = line_id
        self.second_depart_name = second_depart_name
        self.cluster_idx = cluster_idx
        self.doctor_id = doctor_id

    def __str__(self):
        return "\"line_id\":{},\"second_depart_name\":{},\"current_idx\":{},\"doctor_id\":{}".format(self.line_id,
                                                                                                     "\"" + self.second_depart_name + "\"",
                                                                                                     self.cluster_idx,
                                                                                                     self.doctor_id)


def print_similar_array():
    # 打印相似矩阵到文件
    similar_df = codecs.open("sentence_similar_array.txt", 'w', encoding="utf8")
    for line in sentence_similar_array:
        for sim_value in line:
            similar_df.writelines(str(sim_value) + "  ")
        similar_df.writelines("\n")
    similar_df.close()


def compute_cluster_similar_dict(index_):
    cluster_idx = -1
    for goodat_cluster in goodat_cluster_list:
        cluster_idx += 1
        centence_num = 0
        weight = 0
        for cluster in goodat_cluster:
            centence_num += 1
            weight += sentence_similar_array[cluster.line_id][index_]

        if centence_num > 0:
            cluster_similar_dict[cluster_idx] = weight / centence_num
        else:
            cluster_similar_dict[cluster_idx] = 0
    return cluster_similar_dict


def yuCaozuo(arr1, arr2):
    # a1 = re.sub("['[\]'\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", str(arr1))
    # a2 = re.sub("['[\]'\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", str(arr2))
    # return list(str(int(a1) & int(a2)))
    l1 = arr2.__len__()
    l2 = arr1.__len__()
    if l1 == 0 or l2 == 0 or l1 != l2:
        return []
    res = [0] * l1
    for index in range(l2):
        res[index] = (1 if arr1[index] > 0 else 0) & (1 if arr2[index] > 0 else 0)
    return res


def huoCaozuo(arr1, arr2):
    # a1 = re.sub("['[\]'\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", str(arr1))
    # a2 = re.sub("['[\]'\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", str(arr2))
    # return list(str(int(a1) | int(a2)))
    l1 = arr2.__len__()
    l2 = arr1.__len__()
    if l1 == 0 or l2 == 0:
        return []
    res = [0] * l1
    for index in range(l2):
        res[index] = int(arr1[index]) | int(arr2[index])
    return res


def print_goodat_cluster():
    goodat_cluster_write = codecs.open("goodat_cluster.txt", 'w', encoding="utf8")
    goodat_str = "["
    for goodat_cluster in goodat_cluster_list:
        goodat_str += "["
        for cluster in goodat_cluster:
            goodat_str += "{" + cluster.__str__() + "},"
        # 去掉最后一个逗号
        goodat_str = goodat_str[:-1] + "],\n"
    goodat_str = goodat_str[:-2] + "]"
    goodat_cluster_write.writelines(goodat_str)
    goodat_cluster_write.close()


# 打印出医生id和类别的结果  {176910113695:0,174910080032:1,}
def print_docid_class_dict():
    docid_class_write = codecs.open("docid_class_write.txt", 'w', encoding="utf8")
    docid_class_write.writelines("{")
    for goodat_cluster in goodat_cluster_list:
        cluster_str = ""
        for cluster in goodat_cluster:
            docid_class_dict[cluster.doctor_id] = cluster.cluster_idx
            cluster_str = cluster_str + str(cluster.doctor_id) + ":" + str(cluster.cluster_idx) + ","
        docid_class_write.writelines(cluster_str + "\n")
    docid_class_write.writelines("}")
    docid_class_write.close()


# 按照聚类结果，将问诊信息归类
def diaginfo_classification():
    diag_info = pd.read_csv('data/diag.csv')
    diag_info = diag_info.drop_duplicates()
    disease_desc_list = diag_info['disease_desc'].values
    doctor_id_list = diag_info['reception_doctor_id'].values
    write_ = codecs.open("diag_corpus_write.txt", 'w', encoding="utf8")
    write_.writelines("class,doctor_id,disease_desc\n")
    for line_idx in range(doctor_id_list.__len__()):
        if doctor_id_list[line_idx] is None or doctor_id_list[line_idx] == "" or doctor_id_list[
            line_idx] == 173410069661:
            continue
        disease_desc = disease_desc_list[line_idx]
        disease_desc = re.sub(regstr, "", disease_desc)
        if disease_desc is not None and disease_desc != "":
            if docid_class_dict.__contains__(doctor_id_list[line_idx]):
                class_ = docid_class_dict[doctor_id_list[line_idx]]
                desc_class = str(class_) + "," + str(doctor_id_list[line_idx]) + "," + disease_desc + "\n"
                write_.writelines(desc_class)
    write_.close()


# 统计问诊种类-总数的关系
def statistic_diagclass_total():
    diaginfo_class = pd.read_csv('diag_corpus_write.txt')
    diaginfo_class_list = diaginfo_class['class'].values
    class_total_dict = {}
    for class_ in diaginfo_class_list:
        if class_total_dict.__contains__(class_):
            class_total_dict[class_] = class_total_dict[class_] + 1
        else:
            class_total_dict[class_] = 1
    class_total_dict = sorted(class_total_dict.items(), key=operator.itemgetter(1), reverse=True)
    write_ = codecs.open("diagclass_total_write.txt", 'w', encoding="utf8")
    write_.writelines("class,total\n")
    for tuple_ in class_total_dict:
        write_.writelines(str(tuple_[0]) + "," + str(tuple_[1]) + "\n")
    write_.close()


# 按照diagclass_total_write的结果统计每类的结果
def assemble_train_corpus():
    diagclass_total = pd.read_csv('diagclass_total_write.txt')
    diagclass_total_list = diagclass_total['total'].values
    diagclass_class_list = diagclass_total['class'].values
    corpus_class_size = diagclass_total_list[int(diagclass_total_list.__len__() * 0.5)]
    corpus_class_size = 1500
    print("每一类的最大个数为:{}".format(corpus_class_size))
    train_class_total = 28
    print("指定训练的类的个数：{}".format(train_class_total))
    class_trainclass_dict = {}
    trainclass_idx = 0
    for _class_ in diagclass_class_list:
        if trainclass_idx == train_class_total:
            break
        class_trainclass_dict[_class_] = trainclass_idx
        trainclass_idx += 1

    # 将字典写到文件中
    write_ = codecs.open("class_trainclass_write.txt", 'w', encoding="utf8")
    write_.writelines("class,trainclass\n")
    for d_k in class_trainclass_dict:
        write_.writelines(str(d_k) + "," + str(class_trainclass_dict[d_k]) + "\n")
    write_.close()

    diag_corpus = pd.read_csv('diag_corpus_write.txt')
    class_ = diag_corpus["class"]
    doctor_id = diag_corpus["doctor_id"]
    disease_desc = diag_corpus["disease_desc"]
    write_ = codecs.open("train_corpus_write.txt", 'w', encoding="utf8")
    write_.writelines("class,train_class,doctor_id,disease_desc\n")
    type_total_dict = {}
    for index in range(diag_corpus.__len__()):
        if not class_trainclass_dict.__contains__(class_[index]):
            continue
        if type_total_dict.__contains__(class_[index]):
            type_total_dict[class_[index]] = type_total_dict[class_[index]] + 1
        else:
            type_total_dict[class_[index]] = 1
        if type_total_dict[class_[index]] < corpus_class_size:
            write_.writelines(str(class_[index]) + "," + str(class_trainclass_dict[class_[index]]) + "," + str(
                doctor_id[index]) + "," + str(disease_desc[index]) + "\n")
    write_.close()


# 读取停用词
def read_stop_words():
    # 分词
    # step 1 读取停用词
    stop_words = []
    with open('data/stop_words.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    return stop_words


def write_cluster_feature_words():
    # 打开一个json文件
    write_ = codecs.open("feature_words_write.txt", 'w', encoding="utf8")
    with open("./goodat_cluster.txt", 'r', encoding='utf-8') as load_f:
        strF = load_f.read()
        if len(strF) > 0:
            goodat_list = json.loads(strF)
        for good_at_arr in goodat_list:
            line_list = []
            for good_at in good_at_arr:
                line_list += corpus_list[good_at['line_id']].split(" ")
            write_.writelines(str(good_at_arr[0]['current_idx']) + str(set(line_list)) + "\n")
    write_.close()

    # write_ = codecs.open("feature_words_write.txt", 'w', encoding="utf8")
    # write_.writelines("[")
    # for onehot_array in feature_onehot_list:
    #     write_.writelines("[")
    #     feature_list = []
    #     for onehot_index in range(onehot_array.__len__()):
    #         if onehot_array[onehot_index] != 0:
    #             feature_list.append(corpus_id2word[onehot_index] + ",")
    #     write_.writelines(feature_list)
    #     write_.writelines("],\n")
    # write_.writelines("]")
    write_.close()


def read_doctor_corpus():
    df_doctor_info = pd.read_csv('data/doctor.csv')
    df_doctor_info = df_doctor_info.drop_duplicates()
    df_doctor_info = df_doctor_info.loc[0:6000]  # 切片
    doc_goodat_list = df_doctor_info['doc_goodat'].values
    second_depart_name_list = df_doctor_info['second_depart_name'].values
    depart_name_list = []
    for depart_name in second_depart_name_list:
        depart_name = re.sub(regstr, "", depart_name)
        jieba.add_word(depart_name)
        depart_name_list.append(depart_name)
    second_depart_name_list = np.array(depart_name_list)

    doc_id_list = df_doctor_info['doc_id'].values
    goodat_depart = doc_goodat_list + second_depart_name_list + second_depart_name_list + second_depart_name_list
    return doc_goodat_list, second_depart_name_list, doc_id_list, goodat_depart


def write_doc_goodat_tokens(goodat_depart):
    target = codecs.open("med.zh.seg.txt", 'w', encoding="utf8")
    line_num = 1
    for line in goodat_depart:
        line_num = line_num + 1
        line = re.sub(regstr, "", line)
        raw_sentence = []
        raw_words = list(jieba.cut(line))
        for word in raw_words:
            if word not in stop_words and word not in ['qingkan520', 'www', 'com', 'http']:
                raw_sentence.append(word)
        line_seg = " ".join(raw_sentence)
        target.writelines(line_seg + "\n")
    target.close()


def read_doc_goodat_tokens():
    corpus_tockens_list = []
    with open('med.zh.seg.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            corpus_list.append(line[:-1])
            line_list = line.split(" ")
            for word in line_list:
                corpus_tockens_list.append(word.replace("\n", ""))
            line = f.readline()
    return corpus_tockens_list, corpus_list


def write_cluster_process():
    cluster_idx = 0
    corpus_size = corpus_list.__len__()
    process_record = codecs.open("process_record.txt", 'w', encoding="utf8")
    for index_ in range(corpus_size):
        if goodat_cluster_list.__len__() == 0:
            goodat_cluster_list.append(
                [GoodAtCluster(index_, second_depart_name_list[index_], cluster_idx, doc_id_list[index_], )])
        else:
            # 组装每一类特征词语的onehot编码
            # 计算当前行与每一类的相似度
            cluster_similar_dict_1 = compute_cluster_similar_dict(index_)

            # 对dict排序
            cluster_similar_dict_1 = sorted(cluster_similar_dict_1.items(),
                                            key=operator.itemgetter(1), reverse=True)

            current_idx = cluster_similar_dict_1[0][0]
            current_weight = cluster_similar_dict_1[0][1]

            if current_weight > 0.2:
                print("第{}行加入第{}类,相似度:{}".format(index_, current_idx, current_weight))
                process_record.writelines("第{}行加入第{}类,相似度:{}".format(index_, current_idx, current_weight) + "\n")
                goodat_cluster_list[current_idx].append(
                    GoodAtCluster(index_, second_depart_name_list[index_], current_idx, doc_id_list[index_]))
                process_record.writelines("===============\n")
            else:
                cluster_idx += 1
                goodat_cluster_list.append(
                    [GoodAtCluster(index_, second_depart_name_list[index_], cluster_idx, doc_id_list[index_])])
                print("第{}行，新增第{}类".format(index_, cluster_idx))
                process_record.writelines("第{}行，新增第{}类".format(index_, cluster_idx) + "\n")
                process_record.writelines("===============\n")

    process_record.close()


if __name__ == '__main__':
    # 分词
    # step 1 读取停用词
    stop_words = read_stop_words()
    print('step 1 ->停用词读取完毕，共{n}个单词'.format(n=len(stop_words)))

    # step 2 读取语料
    print('step 2 ->读取医生擅长语料，准备聚类')
    doc_goodat_list, second_depart_name_list, doc_id_list, goodat_depart = read_doctor_corpus()

    print("step 3 ->打印医生擅长语料分词结果，生成中间文件：med.zh.seg.txt")
    write_doc_goodat_tokens(goodat_depart)

    # step 3 读取分词后的数据
    print("step 4 ->读取医生擅长语料分词结果,组装成一个大数组，用于后续词频统计等")
    corpus_tockens_list, corpus_list = read_doc_goodat_tokens()

    # step 5 统计词频
    print("step 5 ->开始进行统计词频")
    # wd = Counter(corpus_tockens_list)
    # print(wd.most_common(1000)) # 打印前1000个词

    # step 6 对语料进行one-hot 编码
    # tokenizer = Tokenizer(num_words=int(wd.__len__() * 0.95))  # i创建一个分词器（tokenizer），设置为只考虑前1000个最常见的单词
    # tokenizer.fit_on_texts(corpus_list)  # 构建索引单词
    # sequences = tokenizer.texts_to_sequences(corpus_list)  # 将字符串转换为整数索引组成的列表
    # one_hot_results = tokenizer.texts_to_matrix(corpus_list, mode='binary')  # 可以直接得到one-hot二进制表示。这个分词器也支持除
    # one_hot_array = one_hot_results.tolist()
    # array_list = one_hot_results.tolist()

    print("step 6 ->进行词频统计相关，组装onehot编码,id-word,tf-idf词频矩阵")
    # tf-idf 逆向文档概率
    vectorizer = CountVectorizer()  # 将文本中的词语转换为词频矩阵
    transformer = TfidfTransformer()
    one_hot_results = vectorizer.fit_transform(corpus_list)  # 计算词语出现的次数 3x7 sparse matrix
    one_hot_array = one_hot_results.toarray()
    one_hot_array_similar = one_hot_results.toarray()
    tf_idf_weight = transformer.fit_transform(one_hot_results)  # 将词频矩阵X统计成TF-IDF值
    corpus_id2word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    corpus_word2id = vectorizer.vocabulary_
    tfidf_weight_checkpoint = sorted(tf_idf_weight.data)[int(tf_idf_weight.data.__len__() * 0.5)]
    # print(tfidf_weight_checkpoint)
    # print(corpus_id2word[0])
    # print(tf_idf_weight.toarray())
    # print(sorted(tf_idf_weight.data))

    # 一、词向量模型
    print("step 7 ->开始加载词向量矩阵")
    # cn_model = KeyedVectors.load_word2vec_format('data/sgns.zhihu.bigram', binary=False)
    print("step 7 ->词向量矩阵加载完毕")

    # 将每一行onehot编码按照词语之间的相似度进行赋值
    print("step 8 ->根据句子onehot相似矩阵，计算句子之间的相似度，这个过程比较费时")
    # one_hot_array_similar = assemble_onehot_similar_array()
    # 计算矩阵余弦相似度
    print("step 9 ->根据句子onehot相似矩阵，计算句子之间的相似度")
    A_sparse = sparse.csr_matrix(one_hot_array_similar)
    sentence_similar_array = cosine_similarity(A_sparse).tolist()

    # 打印相似矩阵到文件
    print_similar_array()

    print("step 10 ->根据句子onehot矩阵，计算句子之间的相似度")
    # 计算one-hot编码的余弦相似度
    # A_sparse_onehot = sparse.csr_matrix(one_hot_array)
    # similarities_onehot = cosine_similarity(A_sparse_onehot)

    print("step 11 ->开始文本聚类，打印聚类执行过程,生成中间文件:process_record.txt")

    write_cluster_process()

    # 打印聚类结果
    print("step 11 ->按照医生擅长聚类完毕！")
    print_goodat_cluster()

    print("step 11 ->打印每一类的特征词")
    write_cluster_feature_words()

    print("step 12 ->按照医生id作为key,类别作为value组装字段,生成文件:docid_class_write.txt")
    print_docid_class_dict()

    print("step 13 ->将问诊信息按照上述聚类结果分类，整合训练语料，生成文件：diag_corpus_write.txt")
    # 按照聚类结果将diag.csv 问诊信息进行分类
    diaginfo_classification()

    print("step 14 ->将问诊语料各类的条数进行统计，保证各类的语料不倾斜。生成文件：diagclass_total_write.txt")
    statistic_diagclass_total()

    print("step 15 ->按照上面统计结果，组装训练所需要的语料。生成文件：train_corpus_write.txt")
    assemble_train_corpus()

# also can output sparse matrices
# similarities_sparse = cosine_similarity(A_sparse, dense_output=False)
