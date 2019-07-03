# -*- coding: utf-8 -*-
# Created by Jinkey on 2018/1/4.
__author__ = 'Jinkey'

import tensorflow as tf
import jieba as jb
import numpy as np
import pandas as pd
from classification_demo import utils

titles = utils.load_data(catalogue=utils.MULTI_FLAG)
target = utils.load_label(catalogue=utils.MULTI_FLAG)

titles = []

corpus_list = pd.read_csv("train_corpus_write.txt")
class_list = corpus_list["train_class"]
disease_desc_list = corpus_list["disease_desc"]
target_size = pd.read_csv("class_trainclass_write.txt").__len__()
target = np.zeros(shape=[corpus_list.__len__(), target_size])
for index in range(corpus_list.__len__()):
    titles.append(disease_desc_list[index])
    type_ = class_list[index]
    target[index][type_] = 1

max_sequence_length = 30
embedding_size = 50

# 标题分词
titles = [".".join(jb.cut(t, cut_all=True)) for t in titles]

# word2vec 词袋化
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=1)
text_processed = np.array(list(vocab_processor.fit_transform(titles)))

# 读取标签
dict = vocab_processor.vocabulary_._mapping
sorted_vocab = sorted(dict.items(), key=lambda x: x[1])

# 配置网络结构
model = utils.build_netword(catalogue=utils.MULTI_FLAG, dict=dict, embedding_size=embedding_size,
                            max_sequence_length=max_sequence_length, dense_size=target_size)

# # 训练模型
model.fit(text_processed, target, batch_size=512, epochs=100, )
# # 保存模型
model.save("health_and_tech_design.h5")

# 加载预训练的模型
model.load_weights("health_and_tech_design.h5")

# 预测样本
sen = "您好肚子疼很难受谢谢医生了"
sen_prosessed = " ".join(jb.cut(sen, cut_all=True))
sen_prosessed = vocab_processor.transform([sen_prosessed])
sen_prosessed = np.array(list(sen_prosessed))
result = model.predict(sen_prosessed)

catalogue = list(result[0]).index(max(result[0]))
if max(result[0]) > 0.8:
    if catalogue == 0:
        print("第一类")
    elif catalogue == 1:
        print("第二类")
    elif catalogue == 2:
        print("第三类")
    elif catalogue == 3:
        print("第四类")
    elif catalogue == 4:
        print("第五类")
