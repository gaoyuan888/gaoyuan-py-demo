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
    df_doctor_info = df_doctor_info.loc[0:300]  # 切片
    doc_goodat_list = df_doctor_info['doc_goodat'].values
    second_depart_name_list = df_doctor_info['second_depart_name'].values
    target = codecs.open("med.zh.seg.txt", 'w', encoding="utf8")
    line_num = 1
    for line in doc_goodat_list:
        line_num = line_num + 1
        print('---- processing ', line_num, ' article----------------')
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", line)
        raw_sentence = []
        raw_words = list(jieba.cut(line))
        for word in raw_words:
            if word not in stop_words and word not in ['qingkan520', 'www', 'com', 'http']:
                raw_sentence.append(word)
        line_seg = " ".join(raw_sentence)
        target.writelines(line_seg + "\n")

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
                samples_word_list.append(word)
            line = f.readline()

    # step 4.获得所有tokens的长度
    num_tokens_size = [len(tokens) for tokens in samples]
    num_tokens_size = np.array(num_tokens_size)
    num_tokens_size = np.mean(num_tokens_size) + 2 * np.std(num_tokens_size)

    # step 5 统计词频
    wd = Counter(samples_word_list)
    print(wd.most_common(1000))

    # step 6 对语料进行one-hot 编码
    tokenizer = Tokenizer(num_words=150)  # i创建一个分词器（tokenizer），设置为只考虑前1000个最常见的单词
    tokenizer.fit_on_texts(samples)  # 构建索引单词
    sequences = tokenizer.texts_to_sequences(samples)  # 将字符串转换为整数索引组成的列表
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')  # 可以直接得到one-hot二进制表示。这个分词器也支持除

    # 一、词向量模型
    cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', binary=False)
    array_list = one_hot_results.tolist()
    row_index = -1
    for line_row in array_list:
        row_index = row_index + 1
        print(row_index)
        column_index = 0
        for value in line_row:
            column_index = column_index + 1
            if value == 0:
                # 如果为0，则查找当前行语义最接近的得分补上
                # 定位当前的词语
                word = tokenizer.index_word[column_index]
                # 找到当前行的词语
                tmp_list = samples[row_index].split(" ")
                tmp_max_similary_score = 0
                # 计算分数，找到最大的分数
                for tmp_word in tmp_list:
                    try:
                        similary_score = cn_model.similarity(word, tmp_word)
                        if similary_score > tmp_max_similary_score:
                            tmp_max_similary_score = similary_score
                    except KeyError:
                        pass
                if tmp_max_similary_score > 0.7:
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

    # 组织一个map 进行分组
    cluster_tuple_list = []
    samples_size = samples.__len__()
    index_type = 0
    for index_ in range(samples_size):
        sentence_index = index_ + 1
        if cluster_tuple_list.__len__() == 0:
            idx_dep = (sentence_index, second_depart_name_list[index_],)
            cluster_tuple_list.append((idx_dep,))
            index_type = index_type + 1
        else:
            cluster_tuple_index = -1
            cluster_tuple_dict_tmp = {}
            for tup in cluster_tuple_list:
                cluster_tuple_index = cluster_tuple_index + 1
                # 遍历元组中的行号
                similarity_num = 0
                total_num = 0
                for tup_index in tup:
                    total_num = total_num + 1
                    if similarities[tup_index[0]][index_] > 0.1:
                        similarity_num = similarity_num + 1

                similarity_percent = similarity_num / total_num
                cluster_tuple_dict_tmp[cluster_tuple_index] = similarity_percent

            cluster_tuple_index = -1
            cluster_same_dict_tmp = {}
            for tup in cluster_tuple_list:
                cluster_tuple_index = cluster_tuple_index + 1
                cluster_list = []
                for tup_index in tup:
                    cluster_list.append(one_hot_results.tolist()[tup_index[0]])
                current_onehot = one_hot_results.tolist()[index_]
                same_num_list = [0] * one_hot_results.tolist()[0].__len__()
                for one_host in cluster_list:
                    for ids in range(one_host.__len__()):
                        same_num_list[ids] = 1 if int(one_host[ids]) & int(current_onehot[ids]) == 1 else same_num_list[ids]
                cluster_same_dict_tmp[cluster_tuple_index] = sum(same_num_list)

            # if index_==36 or index_==37 or index_==38:
            #     print(cluster_same_dict_tmp)

            dict_list = sorted(cluster_tuple_dict_tmp.items(), key=operator.itemgetter(1), reverse=True)
            idx_dep = (sentence_index, second_depart_name_list[index_],)
            # 判断当前要归类的节点和之前分组是否包含相同的词语，如果包含则加入，如果不包含，则新区分一个分组
            if dict_list[0][1] > 0.5 :
                cluster_tuple_list[dict_list[0][0]] = cluster_tuple_list[dict_list[0][0]] + (idx_dep,)
            else:
                cluster_tuple_list.append((idx_dep,))
                index_type = index_type + 1

    print(cluster_tuple_list)

# also can output sparse matrices
# similarities_sparse = cosine_similarity(A_sparse, dense_output=False)
