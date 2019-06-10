# coding=gbk
# %matplotlib inline
# gensim��������Ԥѵ��word vector
import warnings
# gensim��������Ԥѵ��word vector
from gensim.models import KeyedVectors
import jieba  # ��ͷִ�
import numpy as np
import operator
warnings.filterwarnings("ignore")

# һ��������ģ��
cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', binary=False)

# step 1 ��ȡͣ�ô�
stop_words = []
with open('stop_words.txt', encoding='utf-8') as f:
    line = f.readline()
    while line:
        stop_words.append(line[:-1])
        line = f.readline()
stop_words = set(stop_words)
print('ͣ�ôʶ�ȡ��ϣ���{n}������'.format(n=len(stop_words)))

# step2 ��ȡ�ı���Ԥ�����ִʣ��õ��ʵ�
raw_word_list = []
sentence_list = []
# ����һ���ֵ��ȡ����
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
        if len(line) > 0:  # ������ӷǿ�
            raw_words = set(jieba.cut(line, cut_all=False))
            dealed_words = []
            for word in raw_words:
                if word not in stop_words and word not in ['qingkan520', 'www', 'com', 'http']:
                    raw_word_list.append(word)
                    dealed_words.append(word)
            sentence_list.append(dealed_words)
            # �Աȴ������ƶȡ���װ�ֶβ�����
            if cluster_dict.__len__() == 0:
                list_tmp = []
                list_tmp.append(dealed_words)
                cluster_dict[dict_id] = list_tmp
                dict_id = dict_id + 1
            else:
                # �����ֵ䣬�Աȴ�������ƶȣ�
                # ����ĳ���ֵ�key�µ�list������һ��������list���еĴ���д���ĳ�������Ĵ������ƶ�����0.6������Ϊ���������Ƶ�
                add_flag = False
                # �����ֵ����ưٷֱ�
                same_num_dict = {}
                similar_percent_dict = {}
                decision_percent_dict = {}
                for key, value in cluster_dict.items():
                    print(key, ' value : ', value)
                    # ����һ������������¼���ƶ�
                    similar_list = []
                    sum_value = 0
                    similar_num_total = 0
                    # ��ʱ��value ��list of list ��ʽ
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

                    # �������ƶȴ���0.6�ĸ������������һ�����ϣ�����Ϊ��������������һ��ģ�����ӵ���ǰ��key�У�����Ļ�����µ��ֵ���
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
                    # ˵��û����ͬ�ģ����¿���һ���ֵ��¼
                    list_tmp = []
                    list_tmp.append(dealed_words)
                    cluster_dict[dict_id] = list_tmp
                    dict_id = dict_id + 1
                else:
                    cluster_dict[same_list[0][0]].append(dealed_words)

        line = f.readline()

print(cluster_dict.__len__())

# 1.������ά�ȣ�һ��������300��ά��������ʾ��embedding_dim=300
embedding_dim = cn_model['ɽ����ѧ'].shape[0]
print('����������{}', format(embedding_dim))
array = cn_model['ɽ����ѧ']
# ��ӡ������
print("array", array)

# 2.���ƶȣ��������ƶ�
similar = cn_model.similarity('Ӥ�׶�', 'Ӥ�׶�')
print("similar", similar)

# �������ƶȼ��㷽��    ���������/����������/������  dot(["����"]/|["����"],["����"]/|["����"])
similar_ = np.dot(cn_model["����"] / np.linalg.norm(cn_model["����"]), cn_model["����"] / np.linalg.norm(cn_model["����"]))
print("similar_", similar_)

# 3.Ѱ�Ҳ���ͬһ��Ĵ���
test_words = "������ Ӥ�� ��ͯ ����"
test_words_res = cn_model.doesnt_match(test_words.split())
print('��' + test_words + "�У�����ͬһ��Ĵ�Ϊ��%s" % test_words_res)

# 4.�ҳ�����Ĵ���������ƶ�
similar_1 = cn_model.most_similar(positive=['����'], topn=10)
print("similar_1", similar_1)

# 5.��һ�������ڴ������е�����
index1 = cn_model.vocab["������"].index
print("index1", index1);

# 6.���������������Ĵ���  215 ��Ӧ�Ĵ�Ϊ��ʦ
word = cn_model.index2word[215];
print("word", word);
