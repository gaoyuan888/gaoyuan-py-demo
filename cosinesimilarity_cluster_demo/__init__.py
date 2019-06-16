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

# 语料id-word字典
corpus_id2word = []
# 句子相似度矩阵
sentence_similarities_array = []
# 每一类特征的onehot编码
feature_onehot_list = []


def assemble_onehot_similar_array():
    row_idx = -1
    for one_hot in one_hot_array_similar:
        row_idx += 1
        # column_index = 0
        column_index = -1
        for value in one_hot:
            column_index += 1
            if value == 0:
                # 如果为0，则查找当前行语义最接近的得分补上
                # 定位当前的词语
                # word = tokenizer.index_word[column_index]
                word = corpus_id2word[column_index]
                # 找到当前行的词语
                tmp_list = corpus_list[row_idx].split(" ")
                tmp_sum_similary_score = 0
                tmp_sum_num = 1
                # 计算分数，找到最大的分数
                for tmp_word in tmp_list:
                    try:
                        tmp_similary_score = cn_model.similarity(word, tmp_word)
                        if tmp_similary_score > 0.4:
                            tmp_sum_similary_score += cn_model.similarity(word, tmp_word)
                            tmp_sum_num += 1
                    except KeyError:
                        pass
                tmp_score = tmp_sum_similary_score / tmp_sum_num
                if tmp_score > 0.4:
                    one_hot_array_similar[row_idx][column_index - 1] = tmp_score
    return one_hot_array_similar


def print_similar_array():
    # 打印相似矩阵到文件
    similar_df = codecs.open("sentence_similarities_array.txt", 'w', encoding="utf8")
    for line in sentence_similarities_array:
        for sim_value in line:
            similar_df.writelines(str(sim_value) + "  ")
        similar_df.writelines("\n")
    similar_df.close()


def compute_cluster_similar_dict(feature_onehot_list, current_onehot):
    cluster_similar_dict = {}
    for cluster_idx in range(feature_onehot_list.__len__()):
        sentence_similarities_array = cosine_similarity([current_onehot, feature_onehot_list[cluster_idx]])
        cluster_similar_dict[cluster_idx] = sentence_similarities_array[0][1]
    return cluster_similar_dict


def assemble_feature_onehot_list():
    feature_onehot_list = []
    for cluster in goodat_cluster_list:
        onehot_sum_list = [0] * tf_idf_weight.toarray()[0].__len__()
        for tuple_list in cluster:
            t1 = [1 if weight >= 0.255 else 0 for weight in tf_idf_weight.toarray()[tuple_list[0]]]
            onehot_sum_list = huoCaozuo(onehot_sum_list, t1)
        feature_onehot_list.append(onehot_sum_list)
    return feature_onehot_list


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
    df_doctor_info = df_doctor_info.loc[0:200]  # 切片
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

    with open('med.zh.seg.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            corpus_list.append(line[:-1])
            line_list = line.split(" ")
            for word in line_list:
                corpus_tockens_list.append(word.replace("\n", ""))
            line = f.readline()

    # step 4.获得所有tokens的长度
    num_tokens_size = [len(tokens) for tokens in corpus_list]
    num_tokens_size = np.array(num_tokens_size)
    num_tokens_size = np.mean(num_tokens_size) + 2 * np.std(num_tokens_size)

    # step 5 统计词频
    wd = Counter(corpus_tockens_list)
    print(wd.most_common(1000))

    # step 6 对语料进行one-hot 编码
    tokenizer = Tokenizer(num_words=int(wd.__len__() * 0.95))  # i创建一个分词器（tokenizer），设置为只考虑前1000个最常见的单词
    tokenizer.fit_on_texts(corpus_list)  # 构建索引单词
    sequences = tokenizer.texts_to_sequences(corpus_list)  # 将字符串转换为整数索引组成的列表
    one_hot_results = tokenizer.texts_to_matrix(corpus_list, mode='binary')  # 可以直接得到one-hot二进制表示。这个分词器也支持除
    one_hot_array = one_hot_results.tolist()
    array_list = one_hot_results.tolist()

    # tf-idf 逆向文档概率
    vectorizer = CountVectorizer()  # 将文本中的词语转换为词频矩阵
    transformer = TfidfTransformer()
    one_hot_results = vectorizer.fit_transform(corpus_list)  # 计算词语出现的次数 3x7 sparse matrix
    tf_idf_weight = transformer.fit_transform(one_hot_results)  # 将词频矩阵X统计成TF-IDF值
    corpus_id2word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    one_hot_array = one_hot_results.toarray()
    one_hot_array_similar = one_hot_results.toarray()
    print(tf_idf_weight.toarray())
    print(sorted(tf_idf_weight.data))
    weight_checked = sorted(tf_idf_weight.data)[int(tf_idf_weight.data.__len__() * 0.5)]
    print(weight_checked)
    print(corpus_id2word[0])

    # 一、词向量模型
    cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', binary=False)

    # 将每一行onehot编码按照词语之间的相似度进行赋值
    one_hot_array_similar = assemble_onehot_similar_array()

    # 计算矩阵余弦相似度
    A_sparse = sparse.csr_matrix(one_hot_array_similar)
    sentence_similarities_array = cosine_similarity(A_sparse)
    # 打印相似矩阵到文件
    print_similar_array()

    # 计算one-hot编码的余弦相似度
    A_sparse_onehot = sparse.csr_matrix(one_hot_array)
    similarities_onehot = cosine_similarity(A_sparse_onehot)

    cluster_idx = 0
    corpus_size = corpus_list.__len__()
    process_record = codecs.open("process_record.txt", 'w', encoding="utf8")
    for index_ in range(corpus_size):
        if goodat_cluster_list.__len__() == 0:
            goodat_cluster_list.append([(index_, second_depart_name_list[index_], cluster_idx,)])
        else:
            # 组装每一类特征词语的onehot编码
            feature_onehot_list = assemble_feature_onehot_list()
            # 计算当前行与每一类的相似度
            cluster_similar_dict = compute_cluster_similar_dict(feature_onehot_list, one_hot_array[index_])
            # 对dict排序
            cluster_similar_dict = sorted(cluster_similar_dict.items(), key=operator.itemgetter(1), reverse=True)
            # 判断当前要归类的节点和之前分组是否包含相同的词语，如果包含则加入，如果不包含，则新区分一个分组
            flag = True
            for cluster_similar in cluster_similar_dict:
                cluster_idx = cluster_similar[0]  # 每一类id
                cluster_weight = cluster_similar[1]  # 每一类的相似度weight
                similar_words = yuCaozuo(feature_onehot_list[cluster_idx], one_hot_array[index_])
                arr_clss = []
                for idx in range(feature_onehot_list[cluster_idx].__len__()):
                    if feature_onehot_list[cluster_idx][idx] != 0:
                        # arr_clss.append(tokenizer.index_word[idx])
                        arr_clss.append(corpus_id2word[idx])
                process_record.writelines("第{}类单词:{}".format(cluster_idx, arr_clss) + "\n")
                arr_onehot = []
                for idx in range(one_hot_array[index_].__len__()):
                    if one_hot_array[index_][idx] != 0:
                        # arr_onehot.append(tokenizer.index_word[idx])
                        arr_onehot.append(corpus_id2word[idx])
                process_record.writelines("第{}行单词:{}".format(index_, arr_onehot) + "\n")
                same_word_list = []
                for idx in range(similar_words.__len__()):
                    if similar_words[idx] != 0:
                        # same_word_list.append(tokenizer.index_word[idx])
                        same_word_list.append(corpus_id2word[idx])
                process_record.writelines(
                    "第{}行与第{}类的相似度:{}相似词为:{}".format(index_, cluster_idx, cluster_weight, same_word_list) + "\n")
                same_words_count = same_word_list.__len__()
                if (cluster_weight > 0.1 or same_words_count >= 1) and flag:
                    print("第{}行加入第{}类,相似度:{}".format(index_, cluster_idx, cluster_weight))
                    process_record.writelines("第{}行加入第{}类,相似度:{}".format(index_, cluster_idx, cluster_weight) + "\n")
                    idx_dep = (index_, second_depart_name_list[index_], cluster_idx,)
                    goodat_cluster_list[cluster_idx].append(idx_dep)
                    flag = False
                process_record.writelines("===============\n")
            if flag:
                cluster_idx += 1
                idx_dep = (index_, second_depart_name_list[index_], cluster_idx,)
                type_list = [idx_dep]
                goodat_cluster_list.append(type_list)
                print("第{}行，新增第{}类".format(index_, cluster_idx))
                process_record.writelines("第{}行，新增第{}类".format(index_, cluster_idx) + "\n")
                process_record.writelines("===============\n")
    process_record.close()
    print(goodat_cluster_list)

# also can output sparse matrices
# similarities_sparse = cosine_similarity(A_sparse, dense_output=False)
