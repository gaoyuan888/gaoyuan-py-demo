# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import operator
# 一、词向量模型
cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', binary=False)
# import sys
# f = open('a.log', 'a')
# sys.stdout = f
# sys.stderr = f

if __name__ == '__main__':
    # 读取词
    samples = ['种植 牙 牙周病 治疗 修复 及口 内 治疗', '乳腺 肿瘤 及 乳房 整形 领域 的 手术 消化系统 等 常见 肿瘤 的 诊治']

    samples = []
    with open('med.zh.seg.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            samples.append(line[:-1])
            line = f.readline()

    # 对语料进行one-hot 编码
    tokenizer = Tokenizer(num_words=100)  # i创建一个分词器（tokenizer），设置为只考虑前1000个最常见的单词
    tokenizer.fit_on_texts(samples)  # 构建索引单词
    sequences = tokenizer.texts_to_sequences(samples)  # 将字符串转换为整数索引组成的列表
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')  # 可以直接得到one-hot二进制表示。这个分词器也支持除

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
                if tmp_max_similary_score>0.8:
                    array_list[row_index][column_index - 1] = tmp_max_similary_score

    # 计算余弦相似度
    A_sparse = sparse.csr_matrix(array_list)
    similarities = cosine_similarity(A_sparse)

    # 组织一个map 进行分组
    cluster_tuple_list = []
    samples_size = samples.__len__()
    index_type = 0
    for index_ in range(samples_size):
        print("处理行id:", index_)
        if cluster_tuple_list.__len__() == 0:
            cluster_tuple_list.append((index_,))
            index_type = index_type + 1
        else:
            cluster_tuple_index = -1
            cluster_tuple_dict_tmp = {}
            for tup in cluster_tuple_list:
                cluster_tuple_index = cluster_tuple_index + 1
                # 遍历元组中的行号
                similarity_num = 0
                tup_num = 0
                for tup_index in tup:
                    tup_num = tup_num + 1
                    if similarities[tup_index][index_] > 0.2:
                        similarity_num = similarity_num + 1
                cluster_tuple_dict_tmp[cluster_tuple_index] = similarity_num/tup_num

            dict_list = sorted(cluster_tuple_dict_tmp.items(), key=operator.itemgetter(1), reverse=True)
            if dict_list[0][1] > 0.4:
                cluster_tuple_list[dict_list[0][0]] = cluster_tuple_list[dict_list[0][0]] + (index_,)
            else:
                cluster_tuple_list.append((index_,))
                index_type = index_type + 1

    print(cluster_tuple_list)

# also can output sparse matrices
# similarities_sparse = cosine_similarity(A_sparse, dense_output=False)

# print(similarities[2][2])

# print('pairwise sparse output:\n {}\n'.format(np.array(similarities_sparse)))
# print(type(similarities_sparse))
# print(type(similarities))
# staa = np.array(similarities_sparse)
# array_list = str(np.array(similarities_sparse)).split('\n')

# target = codecs.open("similary.txt", 'w', encoding="utf8")
# for line_row in similarities:
#     # print(arr)
#     for colum in line_row:
#         target.writelines(colum)
#
# target.close()
