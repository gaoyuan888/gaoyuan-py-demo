import codecs
import re

import jieba
import jieba.analyse

# step 1 读取停用词
stop_words = []
with open('stop_words.txt',encoding= 'utf-8') as f:
    line = f.readline()
    while line:
        stop_words.append(line[:-1])
        line = f.readline()
stop_words = set(stop_words)
print('停用词读取完毕，共{n}个单词'.format(n=len(stop_words)))

def cut_words(sentence):
    # print sentence
    return " ".join(jieba.cut(sentence)).encode('utf-8')


f = codecs.open('med.zh.txt', 'r', encoding="utf8")
target = codecs.open("med.zh.seg.txt", 'w', encoding="utf8")
print('open files')
line_num = 1
line = f.readline()
while line:
    print('---- processing ', line_num, ' article----------------')
    line = re.sub(" ", "", line)
    raw_sentence=[]
    raw_words=list(jieba.cut(line))
    for word in raw_words:
        if word not in stop_words and word not in ['qingkan520', 'www', 'com', 'http']:
            raw_sentence.append(word)
    # line_seg = " ".join(jieba.cut(line))
    line_seg = " ".join(raw_sentence)
    target.writelines(line_seg)
    line_num = line_num + 1
    line = f.readline()
f.close()
target.close()
exit()
while line:
    curr = []
    for oneline in line:
        # print(oneline)
        curr.append(oneline)
    after_cut = map(cut_words, curr)
    target.writelines(after_cut)
    print('saved', line_num, 'articles')
    exit()
    line = f.readline1()
f.close()
target.close()

# python Testjieba.py
