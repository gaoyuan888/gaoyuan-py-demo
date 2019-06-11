from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing

samples = ['种植 牙 牙周病 治疗 修复 及口 内 治疗', '乳腺 肿瘤 及 乳房 整形 领域 的 手术 消化系统 等 常见 肿瘤 的 诊治']
tokenizer = Tokenizer(num_words=20)  # i创建一个分词器（tokenizer），设置为只考虑前1000个最常见的单词
tokenizer.fit_on_texts(samples)  # 构建索引单词
sequences = tokenizer.texts_to_sequences(samples)  # 将字符串转换为整数索引组成的列表
print(sequences)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')  # 可以直接得到one-hot二进制表示。这个分词器也支持除
print(one_hot_results.tolist())

# one-hot编码外其他向量化模式
word_index = tokenizer.word_index  # 得到单词索引
print('Found %s unique tokens.' % len(word_index))
