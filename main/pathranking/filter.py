import pandas as pd
#处理<e>不为1次的情况
train=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
name=train['entity_name']
ques=train['question']

tagall=[]
with open('label/albert-base-v2_label.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        tagall.append(line)
        line=f.readline()

f1=open('error.txt','w',encoding='utf-8')
num=0
with open('test_ques.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        a=line.count('<e>')
        if a!=1:
            f1.write(str(num))
            f1.write('\t')
            #f1.write(name[num])
            f1.write(tagall[num])
            f1.write('\t')
            f1.write(ques[num])
            f1.write('\t')
            f1.write(line[:-1])
            f1.write('\n')
        line=f.readline()
        num+=1