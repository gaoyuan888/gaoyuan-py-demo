# gensim用来加载预训练word vector
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

regstr = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+|[a-zA-Z0-9_]+"


class GoodAtCluster:
    def __init__(self, line_id, second_depart_name, cluster_idx, doctor_id):
        self.line_id = line_id
        self.second_depart_name = second_depart_name
        self.cluster_idx = cluster_idx
        self.doctor_id = doctor_id

    def __str__(self):
        return "line_id:{},second_depart_name:{},current_idx:{},doctor_id:{}".format(self.line_id,
                                                                                     "'" + self.second_depart_name + "'",
                                                                                     self.cluster_idx,
                                                                                     self.doctor_id).replace(" ",
                                                                                                             "").replace(
            "\t", "")


# 组长one-hot 相似矩阵
def assemble_onehot_similar_array():
    row_idx = -1
    for one_hot in one_hot_array_similar:
        row_idx += 1
        print(row_idx)
        # column_index = 0
        # 计算当前行的词语的词向量矩阵
        word_list = corpus_list[row_idx].split(" ")
        w2v_dim_list = []
        for word in word_list:
            try:
                w2v_dim_list.append(cn_model[word])
            except KeyError:
                pass
        w2v_dim_array = np.array(w2v_dim_list, dtype=np.float)
        column_index = -1
        for value in one_hot:
            column_index += 1
            if value == 0:
                # 如果为0，则查找当前行语义最接近的得分补上
                # 定位当前的词语
                # word = tokenizer.index_word[column_index]
                word = corpus_id2word[column_index]
                # 找到当前行的词语
                tmp_sum_similary_score = 0
                tmp_sum_num = 0
                # 计算分数，找到最大的分数
                try:
                    current_word_dim = cn_model[word]
                    word_similar_array = cn_model.cosine_similarities(current_word_dim, w2v_dim_array).tolist()
                    for weitht_ in word_similar_array:
                        if weitht_ > 0.95:
                            tmp_sum_similary_score += weitht_
                            tmp_sum_num += 1
                    if tmp_sum_similary_score > 0:
                        tmp_score = tmp_sum_similary_score / tmp_sum_num
                        if tmp_score > 0.95:
                            one_hot_array_similar[row_idx][column_index] = 1
                except KeyError:
                    pass
    # for onehot_index in one_hot_array_similar:
    #     onehot2word_list = []
    #     for onehot in range(onehot_index.__len__()):
    #         if onehot_index[onehot] != 0 and onehot_index[onehot] != 1:
    #             onehot2word_list.append(corpus_id2word[onehot] + "+")
    #         elif onehot_index[onehot] == 1:
    #             onehot2word_list.append(corpus_id2word[onehot])
    #     print(onehot2word_list)
    return one_hot_array_similar


def print_similar_array():
    # 打印相似矩阵到文件
    similar_df = codecs.open("sentence_similar_array.txt", 'w', encoding="utf8")
    for line in sentence_similar_array:
        for sim_value in line:
            similar_df.writelines(str(sim_value) + "  ")
        similar_df.writelines("\n")
    similar_df.close()


def compute_cluster_similar_dict(feature_onehot_list, current_onehot):
    # cluster_similar_dict = {}
    # cluster_similar_mult_samenum_dict = {}
    # cluster_same_dict = {}

    for cluster_idx in range(feature_onehot_list.__len__()):
        sentence_similar_1 = cosine_similarity([current_onehot, feature_onehot_list[cluster_idx]])
        # 计算当前行与当前类特征的相同词
        similar_words_1 = yuCaozuo(feature_onehot_list[cluster_idx], current_onehot)
        cluster_similar_dict[cluster_idx] = sentence_similar_1[0][1]
        cluster_similar_mult_samenum_dict[cluster_idx] = sentence_similar_1[0][1] * sum(similar_words_1)
        cluster_same_dict[cluster_idx] = similar_words_1

    return cluster_similar_dict, cluster_similar_mult_samenum_dict, cluster_same_dict


# 组装每一类one-hot特征编码
# def assemble_feature_onehot_list():
#     feature_onehot_list = []
#     for cluster in goodat_cluster_list:
#         onehot_sum_list = [0] * tf_idf_weight.toarray()[0].__len__()
#         for tuple_list in cluster:
#             t1 = [1 if weight >= tfidf_weight_checkpoint else 0 for weight in tf_idf_weight.toarray()[tuple_list[0]]]
#             onehot_sum_list = huoCaozuo(onehot_sum_list, t1)
#         feature_onehot_list.append(onehot_sum_list)
#     return feature_onehot_list


# 组装每一类one-hot特征编码
# def assemble_feature_onehot_list():
#     feature_onehot_list = []
#     for cluster in goodat_cluster_list:
#         feature_words_list = []
#         for tuple_list in cluster:
#             feature_words_list += corpus_list[tuple_list.line_id].split(" ")
#         # 计算词频，取前百分之40的词频
#         wd = Counter(feature_words_list)
#         feature_words_list = wd.most_common(int(wd.__len__() * 0.4))
#         # 将最频繁词频转换成one-hot编码
#         feature_onehot = [0] * tf_idf_weight.toarray()[0].__len__()
#         for word in feature_words_list:
#             try:
#                 feature_onehot[corpus_word2id[word[0]]] = 1
#             except KeyError:
#                 pass
#         feature_onehot_list.append(feature_onehot)
#     return feature_onehot_list

# 组装每一类one-hot特征编码
def assemble_feature_onehot_list(current_cluster_index, index_):
    # 将最频繁词频转换成one-hot编码
    feature_onehot = [0] * tf_idf_weight.toarray()[0].__len__()
    flag = feature_onehot_list.__len__() == current_cluster_index

    cluster = goodat_cluster_list[current_cluster_index]
    feature_words_list = []
    num_len_list = []
    for tuple_list in cluster:
        current_cor_list = corpus_list[tuple_list.line_id].split(" ")
        feature_words_list += current_cor_list
        num_len_list.append(set(current_cor_list).__len__())
    # 计算词频，取前百分之40的词频

    wd = Counter(feature_words_list)
    # num_len_list = np.array(num_len_list)
    # len_tmp = int(np.mean(num_len_list))
    feature_words_list = wd.most_common()

    feature_words_disk[current_cluster_index] = feature_words_list

    # feature_words_list = wd.most_common(wd.__len__() if int(wd.__len__() * 0.8) < 3 else int(wd.__len__() * 0.8))
    for feature_idx in feature_words_disk:
        for word in feature_words_disk[feature_idx]:
            try:
                feature_onehot[corpus_word2id[word[0]]] = 1
            except KeyError:
                pass
        if feature_idx == current_cluster_index:
            feature_onehot_list.append(feature_onehot)
        else:
            feature_onehot_list[feature_idx] = feature_onehot

    return feature_onehot_list


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
    goodat_cluster_write.writelines("[")
    for goodat_cluster in goodat_cluster_list:
        cluster_str = "["
        for cluster in goodat_cluster:
            cluster_str = cluster_str + "{" + cluster.__str__() + "},"
        goodat_cluster_write.writelines(cluster_str + "],\n")
    goodat_cluster_write.writelines("]")
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
    corpus_class_size = diagclass_total_list[int(diagclass_total_list.__len__() * 0.5)]
    print("每一类的最大个数为:{}".format(corpus_class_size))
    diag_corpus = pd.read_csv('diag_corpus_write.txt')
    class_ = diag_corpus["class"]
    doctor_id = diag_corpus["doctor_id"]
    disease_desc = diag_corpus["disease_desc"]
    write_ = codecs.open("train_corpus_write.txt", 'w', encoding="utf8")
    write_.writelines("class,doctor_id,disease_desc\n")
    type_total_dict = {}
    for index in range(diag_corpus.__len__()):
        if type_total_dict.__contains__(class_[index]):
            type_total_dict[class_[index]] = type_total_dict[class_[index]] + 1
        else:
            type_total_dict[class_[index]] = 1
        if type_total_dict[class_[index]] < corpus_class_size:
            # print(diag_corpus[index])
            write_.writelines(str(class_[index]) + "," + str(doctor_id[index]) + "," + str(disease_desc[index]) + "\n")
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
    write_ = codecs.open("feature_words_write.txt", 'w', encoding="utf8")
    write_.writelines("[")
    for onehot_array in feature_onehot_list:
        write_.writelines("[")
        feature_list = []
        for onehot_index in range(onehot_array.__len__()):
            if onehot_array[onehot_index] != 0:
                feature_list.append(corpus_id2word[onehot_index] + ",")
        write_.writelines(feature_list)
        write_.writelines("],\n")
    write_.writelines("]")
    write_.close()


def read_doctor_corpus():
    df_doctor_info = pd.read_csv('data/doctor.csv')
    df_doctor_info = df_doctor_info.drop_duplicates()
    df_doctor_info = df_doctor_info.loc[0:50]  # 切片
    doc_goodat_list = df_doctor_info['doc_goodat'].values
    second_depart_name_list = df_doctor_info['second_depart_name'].values
    doc_id_list = df_doctor_info['doc_id'].values
    goodat_depart = doc_goodat_list + second_depart_name_list + second_depart_name_list + second_depart_name_list
    return doc_goodat_list, second_depart_name_list, doc_id_list, goodat_depart


def write_doc_goodat_tokens(goodat_depart):
    target = codecs.open("med.zh.seg.txt", 'w', encoding="utf8")
    line_num = 1
    for line in goodat_depart:
        line_num = line_num + 1
        # print('---- processing ', line_num, ' article----------------')
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
    current_cluster_index = cluster_idx
    for index_ in range(corpus_size):
        if goodat_cluster_list.__len__() == 0:
            goodat_cluster_list.append(
                [GoodAtCluster(index_, second_depart_name_list[index_], cluster_idx, doc_id_list[index_], )])
        else:
            # 组装每一类特征词语的onehot编码
            feature_onehot_list_1 = assemble_feature_onehot_list(current_cluster_index, index_)
            # 计算当前行与每一类的相似度
            cluster_similar_dict_1, cluster_similar_mult_samenum_dict_1, cluster_same_dict_1 = compute_cluster_similar_dict(
                feature_onehot_list,
                one_hot_array_similar[index_]
                # current_cluster_index
            )

            # 对dict排序
            # cluster_similar_dict = sorted(cluster_similar_dict.items(), key=operator.itemgetter(1), reverse=True)
            cluster_similar_mult_samenum_dict_1 = sorted(cluster_similar_mult_samenum_dict_1.items(),
                                                         key=operator.itemgetter(1), reverse=True)

            # 判断当前要归类的节点和之前分组是否包含相同的词语，如果包含则加入，如果不包含，则新区分一个分组
            flag = True
            for cluster_similar in cluster_similar_mult_samenum_dict_1:
                current_idx = cluster_similar[0]  # 每一类id
                # current_weight = cluster_similar[1]  # 每一类的相似度weight
                current_weight = cluster_similar_dict_1[current_idx]  # 每一类的相似度weight
                similar_words = cluster_same_dict_1[current_idx]
                # similar_words = yuCaozuo(feature_onehot_list[current_idx], one_hot_array_similar[index_])
                arr_clss = []
                for idx in range(feature_onehot_list[current_idx].__len__()):
                    if feature_onehot_list[current_idx][idx] != 0:
                        # arr_clss.append(tokenizer.index_word[idx])
                        arr_clss.append(corpus_id2word[idx])
                process_record.writelines("第{}类单词:{}".format(current_idx, arr_clss) + "\n")

                arr_onehot = []
                for idx in range(one_hot_array[index_].__len__()):
                    if one_hot_array[index_][idx] != 0:
                        # arr_onehot.append(tokenizer.index_word[idx])
                        arr_onehot.append(corpus_id2word[idx])
                process_record.writelines("第{}行单词:{}".format(index_, arr_onehot) + "\n")

                # arr_onehot_similary = []
                # for idx in range(one_hot_array_similar[index_].__len__()):
                #     if one_hot_array_similar[index_][idx] != 0:
                #         # arr_onehot.append(tokenizer.index_word[idx])
                #         arr_onehot_similary.append(corpus_id2word[idx])
                # process_record.writelines("第{}行单词(相似):{}".format(index_, arr_onehot_similary) + "\n")

                same_word_list = []
                for idx in range(similar_words.__len__()):
                    if similar_words[idx] != 0:
                        # same_word_list.append(tokenizer.index_word[idx])
                        same_word_list.append(corpus_id2word[idx])
                process_record.writelines(
                    "第{}行与第{}类的相似度:{}，相同词为:{}".format(index_, current_idx, current_weight, same_word_list) + "\n")
                same_words_count = same_word_list.__len__()
                if ((current_weight > 0.13 and same_words_count > 2) or same_words_count >= arr_onehot.__len__() * 0.4) and flag:
                    print("第{}行加入第{}类,相似度:{}".format(index_, current_idx, current_weight))
                    process_record.writelines(
                        "第{}行加入第{}类,相似度:{}".format(index_, current_idx, current_weight) + "\n")
                    goodat_cluster_list[current_idx].append(
                        GoodAtCluster(index_, second_depart_name_list[index_], current_idx, doc_id_list[index_]))
                    # (index_, second_depart_name_list[index_], current_idx, doc_id_list[index_],)
                    current_cluster_index = cluster_idx
                    flag = False
                    process_record.writelines("===============\n")
                    # break
                process_record.writelines("===============\n")
            if flag:
                cluster_idx += 1
                goodat_cluster_list.append(
                    [GoodAtCluster(index_, second_depart_name_list[index_], cluster_idx, doc_id_list[index_])])
                # [(index_, second_depart_name_list[index_], cluster_idx, doc_id_list[index_],)]
                print("第{}行，新增第{}类".format(index_, cluster_idx))
                process_record.writelines("第{}行，新增第{}类".format(index_, cluster_idx) + "\n")
                current_cluster_index = cluster_idx
                process_record.writelines("===============\n")

    process_record.close()


# def write_cluster_process():
#     cluster_idx = 0
#     corpus_size = corpus_list.__len__()
#     process_record = codecs.open("process_record.txt", 'w', encoding="utf8")
#     current_cluster_index = cluster_idx
#     for index_ in range(corpus_size):
#         if goodat_cluster_list.__len__() == 0:
#             goodat_cluster_list.append(
#                 [GoodAtCluster(index_, second_depart_name_list[index_], cluster_idx, doc_id_list[index_], )])
#         else:
#             # 组装每一类特征词语的onehot编码
#             feature_onehot_list = assemble_feature_onehot_list(current_cluster_index)
#             # 计算当前行与每一类的相似度
#             cluster_similar_dict = compute_cluster_similar_dict(feature_onehot_list, one_hot_array[index_])
#             # 对dict排序
#             cluster_similar_dict = sorted(cluster_similar_dict.items(), key=operator.itemgetter(1), reverse=True)
#             # 判断当前要归类的节点和之前分组是否包含相同的词语，如果包含则加入，如果不包含，则新区分一个分组
#             flag = True
#             for cluster_similar in cluster_similar_dict:
#                 current_idx = cluster_similar[0]  # 每一类id
#                 current_weight = cluster_similar[1]  # 每一类的相似度weight
#                 similar_words = yuCaozuo(feature_onehot_list[current_idx], one_hot_array[index_])
#                 arr_clss = []
#                 for idx in range(feature_onehot_list[current_idx].__len__()):
#                     if feature_onehot_list[current_idx][idx] != 0:
#                         # arr_clss.append(tokenizer.index_word[idx])
#                         arr_clss.append(corpus_id2word[idx])
#                 process_record.writelines("第{}类单词:{}".format(current_idx, arr_clss) + "\n")
#                 arr_onehot = []
#                 for idx in range(one_hot_array[index_].__len__()):
#                     if one_hot_array[index_][idx] != 0:
#                         # arr_onehot.append(tokenizer.index_word[idx])
#                         arr_onehot.append(corpus_id2word[idx])
#                 process_record.writelines("第{}行单词:{}".format(index_, arr_onehot) + "\n")
#                 same_word_list = []
#                 for idx in range(similar_words.__len__()):
#                     if similar_words[idx] != 0:
#                         # same_word_list.append(tokenizer.index_word[idx])
#                         same_word_list.append(corpus_id2word[idx])
#                 process_record.writelines(
#                     "第{}行与第{}类的相似度:{}，相同词为:{}".format(index_, current_idx, current_weight, same_word_list) + "\n")
#                 same_words_count = same_word_list.__len__()
#                 if arr_onehot.__len__() > 3:
#                     if ((current_weight > 0.16 and same_words_count >= 1) or same_words_count >= 3) and flag:
#                         print("第{}行加入第{}类,相似度:{}".format(index_, current_idx, current_weight))
#                         process_record.writelines(
#                             "第{}行加入第{}类,相似度:{}".format(index_, current_idx, current_weight) + "\n")
#                         goodat_cluster_list[current_idx].append(
#                             GoodAtCluster(index_, second_depart_name_list[index_], current_idx, doc_id_list[index_]))
#                         # (index_, second_depart_name_list[index_], current_idx, doc_id_list[index_],)
#                         current_cluster_index = cluster_idx
#                         flag = False
#                         process_record.writelines("===============\n")
#                         # break
#                 elif arr_onehot.__len__() <= 3:
#                     if same_words_count >= 1 and flag:
#                         print("第{}行加入第{}类,相似度:{}".format(index_, current_idx, current_weight))
#                         process_record.writelines(
#                             "第{}行加入第{}类,相似度:{}".format(index_, current_idx, current_weight) + "\n")
#                         goodat_cluster_list[current_idx].append(
#                             GoodAtCluster(index_, second_depart_name_list[index_], current_idx, doc_id_list[index_]))
#                         # (index_, second_depart_name_list[index_], current_idx, doc_id_list[index_],)
#                         current_cluster_index = cluster_idx
#                         flag = False
#                         process_record.writelines("===============\n")
#                         # break
#                 process_record.writelines("===============\n")
#             if flag:
#                 cluster_idx += 1
#                 goodat_cluster_list.append(
#                     [GoodAtCluster(index_, second_depart_name_list[index_], cluster_idx, doc_id_list[index_])])
#                 # [(index_, second_depart_name_list[index_], cluster_idx, doc_id_list[index_],)]
#                 print("第{}行，新增第{}类".format(index_, cluster_idx))
#                 process_record.writelines("第{}行，新增第{}类".format(index_, cluster_idx) + "\n")
#                 current_cluster_index = cluster_idx
#                 process_record.writelines("===============\n")
#
#     process_record.close()


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
    sentence_similar_array = cosine_similarity(A_sparse)

    # 打印相似矩阵到文件
    print_similar_array()

    print("step 10 ->根据句子onehot矩阵，计算句子之间的相似度")
    # 计算one-hot编码的余弦相似度
    A_sparse_onehot = sparse.csr_matrix(one_hot_array)
    similarities_onehot = cosine_similarity(A_sparse_onehot)

    print("step 11 ->开始文本聚类，打印聚类执行过程,生成中间文件:process_record.txt")
    write_cluster_process()

    print("step 11 ->打印每一类的特征词")
    write_cluster_feature_words()

    # 打印聚类结果
    print("step 11 ->按照医生擅长聚类完毕！")
    print_goodat_cluster()

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
