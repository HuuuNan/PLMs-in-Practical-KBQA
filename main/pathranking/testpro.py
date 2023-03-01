#!/usr/bin/python3
# coding=gbk
import pandas as pd
import json
import time
import os

MODEL='bert-base-uncased'
SCALE='small'

os.makedirs(SCALE+'/'+MODEL, exist_ok=True)
test=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
name_list=[]
with open('../实体识别/'+MODEL+'_query/query.test','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.split(' %%%% ')
        name_list.append(line[1].strip())
        line=f.readline()

total_dict=dict()
total=0

f=open('test_ques.txt','w',encoding='utf-8')
for i in range(0,len(test)):
    temp=dict()
    ques=test['question'][i]
    name=name_list[i]
    ques1=ques
    tempt=time.time()
    if name==ques:
        total=total+time.time()-tempt  
        f.write(ques1)
        f.write('\n')
        continue
    if name 
    name1=' '+name+' '
    name2=' '+name
    name3=name+' '
    if name1 in ques:
        ques1=ques.replace(name1,' <e> ')
    if name2 in ques:
        ques1=ques.replace(name2,' <e>')
    if name3 in ques:
        ques1=ques.replace(name3,'<e> ')
    total=total+time.time()-tempt
    f.write(ques1)
    f.write('\n')
    
f.close()
print('测试集替换头实体耗时：',total)

with open(SCALE+'/'+MODEL+'/test_dict.json', 'w') as f:
    json.dump(total_dict, f,ensure_ascii=False)
    