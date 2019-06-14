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


# def tf_cluster():
#     cluster_onehot_list = []
#     for cluster in cluster_tuple_list:
#         onehot_sum_list = [0] * one_hot_array[0].__len__()
#         onehot_list = [0] * one_hot_array[0].__len__()
#         onehot_list_bak = one_hot_array[cluster[0][0]]
#         for tuple_list in cluster:
#             onehot_sum_list = np.sum([onehot_sum_list, one_hot_array[tuple_list[0]]], axis=0)
#             onehot_list_bak = yuCaozuo(onehot_list_bak, one_hot_array[tuple_list[0]])
#         for idx in range(one_hot_array[0].__len__()):
#             onehot_list[idx] = 0 if onehot_sum_list[idx] / sum(onehot_sum_list) < 0.1 else 1
#         print(onehot_sum_list)
#         if sum(onehot_list) == 0:
#             cluster_onehot_list.append(onehot_list_bak)
#         else:
#             cluster_onehot_list.append(onehot_list)
#     print(cluster_onehot_list)
#     return cluster_onehot_list

def tf_cluster():
    cluster_onehot_list = []
    for cluster in cluster_tuple_list:
        onehot_sum_list = [0] * one_hot_array[0].__len__()
        for tuple_list in cluster:
            t1 = [1 if weight >= 0.2 else 0 for weight in one_hot_array[tuple_list[0]]]
            onehot_sum_list |= t1
        print(onehot_sum_list)
        cluster_onehot_list.append(onehot_sum_list)
    print(cluster_onehot_list)
    return cluster_onehot_list


def yuCaozuo(arr1, arr2):
    # a1 = re.sub("['[\]'\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", str(arr1))
    # a2 = re.sub("['[\]'\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", str(arr2))
    # return list(str(int(a1) & int(a2)))
    l1 = arr2.__len__()
    l2 = arr1.__len__()
    if l1 == 0 or l2 == 0:
        return []
    res = [0] * l1
    for index in range(l2):
        res[index] = int(arr1[index]) & int(arr2[index])
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


if __name__ == '__main__':

    # 分词
    # step 1 读取停用词
    stop_words = []
    with open('stop_words.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个单词'.format(n=len(stop_words)))

    # step 2 读取语料
    print('open files')
    # df_diag = pd.read_csv('diag.csv', header=1, encoding='utf-8')

    df_doctor_info = pd.read_csv('doctor.csv')
    df_doctor_info = df_doctor_info.loc[0:10]  # 切片
    doc_goodat_list = df_doctor_info['doc_goodat'].values
    second_depart_name_list = df_doctor_info['second_depart_name'].values
    goodat_depart = doc_goodat_list + second_depart_name_list

    target = codecs.open("med.zh.seg.txt", 'w', encoding="utf8")
    line_num = 1
    for line in goodat_depart:
        line_num = line_num + 1
        # print('---- processing ', line_num, ' article----------------')
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", line)
        raw_sentence = []
        raw_words = list(jieba.cut(line))
        for word in raw_words:
            if word not in stop_words and word not in ['qingkan520', 'www', 'com', 'http']:
                raw_sentence.append(word)
        line_seg = " ".join(raw_sentence)
        target.writelines(line_seg + "\n")
    target.close()

    # step 3 读取分词后的数据
    # samples = ['种植 牙 牙周病 治疗 修复 及口 内 治疗', '乳腺 肿瘤 及 乳房 整形 领域 的 手术 消化系统 等 常见 肿瘤 的 诊治']
    samples = []
    samples_word_list = []
    with open('med.zh.seg.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            samples.append(line[:-1])
            line_list = line.split(" ")
            for word in line_list:
                samples_word_list.append(word.replace("\n", ""))
            line = f.readline()

    # step 4.获得所有tokens的长度
    num_tokens_size = [len(tokens) for tokens in samples]
    num_tokens_size = np.array(num_tokens_size)
    num_tokens_size = np.mean(num_tokens_size) + 2 * np.std(num_tokens_size)

    # step 5 统计词频
    wd = Counter(samples_word_list)
    print(wd.most_common(1000))

    # step 6 对语料进行one-hot 编码
    tokenizer = Tokenizer(num_words=int(wd.__len__() * 0.95))  # i创建一个分词器（tokenizer），设置为只考虑前1000个最常见的单词
    tokenizer.fit_on_texts(samples)  # 构建索引单词
    sequences = tokenizer.texts_to_sequences(samples)  # 将字符串转换为整数索引组成的列表
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')  # 可以直接得到one-hot二进制表示。这个分词器也支持除
    one_hot_array = one_hot_results.tolist()
    array_list = one_hot_results.tolist()

    # tf-idf 逆向文档概率
    vectorizer = CountVectorizer()  # 将文本中的词语转换为词频矩阵
    transformer = TfidfTransformer()
    X = vectorizer.fit_transform(samples)  # 计算词语出现的次数 3x7 sparse matrix
    tfidf = transformer.fit_transform(X)  # 将词频矩阵X统计成TF-IDF值
    id_word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    one_hot_array = X.toarray()
    array_list = X.toarray()
    print(tfidf.toarray())
    print(id_word[0])

    # 一、词向量模型
    cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', binary=False)

    row_index = -1
    for line_row in array_list:
        row_index = row_index + 1
        print(row_index)
        # column_index = 0
        column_index = -1
        for value in line_row:
            column_index = column_index + 1
            if value == 0:
                # 如果为0，则查找当前行语义最接近的得分补上
                # 定位当前的词语
                # word = tokenizer.index_word[column_index]
                word = id_word[column_index]
                # 找到当前行的词语
                tmp_list = samples[row_index].split(" ")
                tmp_max_similary_score = 0
                # 计算分数，找到最大的分数
                for tmp_word in tmp_list:
                    try:
                        similary_score = cn_model.similarity(word, tmp_word)
                        tmp_max_similary_score += similary_score
                    except KeyError:
                        pass
                if tmp_max_similary_score / tmp_list.__len__() > 0.2:
                    array_list[row_index][column_index - 1] = tmp_max_similary_score

    # 计算矩阵余弦相似度
    A_sparse = sparse.csr_matrix(array_list)
    similarities = cosine_similarity(A_sparse)

    # 计算one-hot编码的余弦相似度
    A_sparse_onehot = sparse.csr_matrix(one_hot_results.tolist())
    similarities_onehot = cosine_similarity(A_sparse_onehot)

    # 打印相似矩阵到文件
    similar_df = codecs.open("similarities.txt", 'w', encoding="utf8")
    for line in similarities:
        for sim_value in line:
            similar_df.writelines(str(sim_value) + "  ")
        similar_df.writelines("\n")
    similar_df.close()

    # 组织一个map 进行分组
    cluster_tuple_list = []
    cluster_type = 0
    samples_size = samples.__len__()

    process_df = codecs.open("process_df.txt", 'w', encoding="utf8")
    for index_ in range(samples_size):
        # cluster_onehot_list = []
        # for cluster in cluster_tuple_list:
        #     onehot_sum_list = one_hot_array[cluster[0][0]]
        #     for tuple_list in cluster:
        #         onehot_sum_list = yuCaozuo(onehot_sum_list, one_hot_array[tuple_list[0]])
        #     cluster_onehot_list.append(onehot_sum_list)
        cluster_onehot_list = tf_cluster()
        if cluster_tuple_list.__len__() == 0:
            idx_dep = (index_, second_depart_name_list[index_], cluster_type,)
            type_list = [idx_dep]
            cluster_tuple_list.append(type_list)
        else:
            cluster_tuple_dict_tmp = {}
            for tup_list in cluster_tuple_list:
                # 遍历元组中的行号
                similarity_num = 0
                total_num = 0
                for tup_index in tup_list:
                    total_num = total_num + 1
                    if similarities[tup_index[0]][index_] > 0.1:
                        similarity_num = similarity_num + 1
                similarity_percent = similarity_num / total_num
                cluster_tuple_dict_tmp[tup_list[0][2]] = similarity_percent
            dict_list = sorted(cluster_tuple_dict_tmp.items(), key=operator.itemgetter(1), reverse=True)
            # 判断当前要归类的节点和之前分组是否包含相同的词语，如果包含则加入，如果不包含，则新区分一个分组
            flag = True
            for dict in dict_list:
                similar_words = yuCaozuo(cluster_onehot_list[dict[0]], one_hot_array[index_])
                same_words_count = sum(similar_words)
                if dict[1] > 0.5 and same_words_count > 0:
                    # if dict[1] > 0.5:
                    print("第{}行，加入第{}类,相似度:{}".format(index_, dict[0], dict[1]))
                    idx_dep = (index_, second_depart_name_list[index_], dict[0],)
                    cluster_tuple_list[dict[0]].append(idx_dep)
                    flag = False
                    break
            if flag:
                cluster_type += 1
                idx_dep = (index_, second_depart_name_list[index_], cluster_type,)
                type_list = [idx_dep]
                cluster_tuple_list.append(type_list)
                print("第{}行，新增第{}类".format(index_, dict[0]))

            for dict in dict_list:
                similar_words = yuCaozuo(cluster_onehot_list[dict[0]], one_hot_array[index_])
                arr_clss = []
                for idx in range(cluster_onehot_list[dict[0]].__len__()):
                    if cluster_onehot_list[dict[0]][idx] == 1:
                        # arr_clss.append(tokenizer.index_word[idx])
                        arr_clss.append(id_word[idx])
                process_df.writelines("第{}类单词:{}".format(dict[0], arr_clss) + "\n")
                arr_onehot = []
                for idx in range(one_hot_array[index_].__len__()):
                    if one_hot_array[index_][idx] == 1:
                        # arr_onehot.append(tokenizer.index_word[idx])
                        arr_onehot.append(id_word[idx])
                process_df.writelines("第{}行单词:{}".format(index_, arr_onehot) + "\n")
                same_word_list = []
                for idx in range(similar_words.__len__()):
                    if similar_words[idx] == 1:
                        # same_word_list.append(tokenizer.index_word[idx])
                        same_word_list.append(id_word[idx])
                process_df.writelines("第{}行与第{}类的相似词为:{}".format(index_, dict[0], same_word_list) + "\n")
                process_df.writelines("===============\n")

    process_df.close()
    print(cluster_tuple_list)

# also can output sparse matrices
# similarities_sparse = cosine_similarity(A_sparse, dense_output=False)
