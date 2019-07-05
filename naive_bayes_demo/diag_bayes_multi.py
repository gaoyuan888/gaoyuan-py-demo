# coding:utf-8
import pandas as pd
import jieba
import numpy

df_news = pd.read_csv('./train_corpus_write.txt', encoding='utf-8')
# https://www.jianshu.com/p/89de6c085d22
df_news = df_news.dropna()
print(df_news.head())
print(df_news.shape)

# 1.分词：使用结吧分词器 ###
content = df_news.disease_desc.values.tolist()
train_class = df_news.train_class.values.tolist()
print(content[1000])
content_S = []
train_class_S = []
for line_idx in range(content.__len__()):
    current_segment = jieba.lcut(content[line_idx])
    if len(current_segment) > 1 and current_segment != '\r\n':  # 换行符
        content_S.append(current_segment)
        train_class_S.append(train_class[line_idx])
print(content_S[1000])
df_content = pd.DataFrame({'content_S': content_S})
print(df_content.head())
stopwords = pd.read_csv("stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')
print(stopwords.head(20))


def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words


contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)

# df_content.content_S.isin(stopwords.stopword)
# df_content=df_content[~df_content.content_S.isin(stopwords.stopword)]
# df_content.head()

df_content = pd.DataFrame({'contents_clean': contents_clean})
print(df_content.head())
df_all_words = pd.DataFrame({'all_words': all_words})
print(df_all_words.head())
words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({"count": numpy.size})
words_count = words_count.reset_index().sort_values(by=["count"], ascending=False)
print(words_count.head())

from wordcloud import WordCloud
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
wordcloud = WordCloud(font_path="./data/simhei.ttf", background_color="white", max_font_size=80)
word_frequence = {x[0]: x[1] for x in words_count.head(100).values}
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)

# TF-IDF ：提取关键词
import jieba.analyse

index = 2400
print(df_news['disease_desc'][index])
content_S_str = "".join(content_S[index])
print("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))

# LDA ：主题模型
# 格式要求：list of list形式，分词好的的整个语料
from gensim import corpora, models, similarities
import gensim

# http://radimrehurek.com/gensim/
# 做映射，相当于词袋
dictionary = corpora.Dictionary(contents_clean)
corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean]
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)  # 类似Kmeans自己指定K值

# 一号分类结果
print(lda.print_topic(1, topn=5))
for topic in lda.print_topics(num_topics=20, num_words=5):
    print(topic[1])

ssss = df_news['train_class']
print(df_news['train_class'])

df_train = pd.DataFrame({'contents_clean': contents_clean, 'label': train_class_S})
print(df_train.tail())
print(df_train.label.unique())

# label_mapping = {u"汽车": 1, u"财经": 2, u"科技": 3, u"健康": 4, u"体育": 5, u"教育": 6, u"文化": 7, u"军事": 8, u"娱乐": 9, u"时尚": 0}
# tmp = df_train['label']
# df_train['label'] = list(map(lambda x: label_mapping.get(x), df_train['label']))
# df_train['label'] = df_train['label'].map(label_mapping)
print(df_train.head())
print(df_train['label'])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values,
                                                    random_state=12)

# x_train = x_train.flatten()
print(x_train[0][1])
words = []
for line_index in range(len(x_train)):
    try:
        # x_train[line_index][word_index] = str(x_train[line_index][word_index])
        words.append(' '.join(x_train[line_index]))
    except:
        print(line_index, words[line_index])
print(words[0])
print(len(words))

# 贝叶斯分类器
from sklearn.feature_extraction.text import CountVectorizer

texts = ["dog cat fish", "dog cat cat", "fish bird", 'bird']
cv = CountVectorizer()
cv_fit = cv.fit_transform(texts)
print(cv.get_feature_names())
print(cv_fit.toarray())
print(cv_fit.toarray().sum(axis=0))

vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)
vec.fit(words)
sss = vec.transform(words).toarray()
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(vec.transform(words), y_train)

test_words = []
for line_index in range(len(x_test)):
    try:
        # x_train[line_index][word_index] = str(x_train[line_index][word_index])
        test_words.append(' '.join(x_test[line_index]))
    except:
        print(line_index, test_words[line_index])
test_words[0]

print(classifier.score(vec.transform(test_words), y_test))

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase=False)
vectorizer.fit(words)

classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)

print(classifier.score(vectorizer.transform(test_words), y_test))

# Tokenizing text
count_vect = CountVectorizer()
docs_new = ['手指 出血 严重',
            '胃疼 胃部 胀痛']
# using classifier to predict
predicted = classifier.predict(vectorizer.transform(docs_new))


class_trainclass = pd.read_csv("class_trainclass_write.txt")
trainclass_class_dict = {}
trainclass_list = class_trainclass["trainclass"]
class_train = class_trainclass["class"]
for idx in range(trainclass_list.__len__()):
    trainclass_class_dict[trainclass_list[idx]] = class_train[idx]

for text, c in zip(docs_new, predicted):
    for k, v in trainclass_class_dict.items():
        if c == v:
            print(k + "类==>  " + text)

rest = vectorizer.transform(words)
