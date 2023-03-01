import jieba
import json
import re
import string
jieba.load_userdict("user_dict.txt")


continue_words = string.ascii_lowercase+string.digits
def tokenizer(sentence):
    '''
    按照单个字进行分词
    :param sentence:
    :return:
    '''
    temp = ""
    result = []
    for word in sentence:
        if word in continue_words:
            temp += word
        else:
            if len(temp)>0:
                result.append(temp)
                temp = ""
            result.append(word)
    if len(temp)>0:
        result.append(temp)
    return result

f = open("../data/军事百科.json", "r", encoding="utf8")
f_corpus=open("corpus.txt","w",encoding="utf8")

lines = f.readlines()

for line2 in lines:
    kb_json = json.loads(line2.strip())
    desc = kb_json["简介"].strip().replace("\n", "").replace("\r", "")
    f_corpus.write(" ".join(tokenizer(desc)))

    # spec_words = re.findall('["歼"(IX|IV|V?I{0,3})-.*:a-zA-Z0-9]+', desc)
    # # print(spec_words)
    # for word in spec_words:
    #     jieba.add_word(word, tag='nz')
    # # 修改jieba包init.py中正则表达式
    # jieba.re_han_default = re.compile('(.+)', re.U)
    #
    # f_corpus.write(" ".join(list(jieba.cut(desc))) + "\n")