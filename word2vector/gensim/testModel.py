from gensim.models import Word2Vec

en_wiki_word2vec_model = Word2Vec.load('med.zh.text.model')

testwords = ['白内障','小儿','乳化','肺癌','小儿']
for i in range(5):
    res = en_wiki_word2vec_model.most_similar(testwords[i])
    print (testwords[i])
    print (res)