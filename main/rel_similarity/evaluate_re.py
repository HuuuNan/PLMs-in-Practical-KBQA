#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import json

import sys
import os

sys.path.append("..")
from mylogger import mylog
file_name = os.path.basename(__file__)
file_name = file_name[0:len(file_name)-3]

#��������
MODEL = 'gpt2'
SCALE = 'small'

# set logger
my_logger = mylog.log_creater('./log', file_name+'_'+MODEL+'_'+SCALE+'-out')
my_logger.warning("\n new process start  \n")
my_logger.warning(MODEL+'-'+SCALE)


truth=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
question=truth['question']
entity_mid=truth['entity_mid']
relation=[]
for i in truth['relation']:
    i=i.replace('.',' ')
    i=i.replace('_',' ')
    i=i[3:len(i)]
    relation.append(i) 
    
#�ҳ�ѵ�����г��ֵĺ�û���ֵĹ�ϵ��Ӧ������
quessee=[]
quesunsee=[]
tra_re=set()
train=pd.read_table('../mydata/train.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
test=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
for i in train['relation']:
    tra_re.add(i)
tra_re=list(tra_re)
for index,i in enumerate(test['relation']):
    if i in tra_re:
        quessee.append(test['question'][index])
    else:
        quesunsee.append(test['question'][index])

index=0
top1=0
top2=0
top3=0
top4=0
top5=0
seeac=0
unseeac=0
seere=0
unseere=0
num=0

data=json.load(open(SCALE+'/'+MODEL+'/relation.json','r',encoding='utf-8'))
for i in data.items():
    while i[0]!=question[index]:
        index+=1
    rescore=[]
    for j in i[1].items():
        rescore.append([j[0],j[1]])
    rescore.sort(key=lambda t: (t[1]), reverse=True)
    if i[0] in quessee:
        if len(rescore)>0 and rescore[0][0]==relation[index]:
            top1+=1
            top2+=1
            top3+=1
            top4+=1
            top5+=1
            seeac+=1
            seere+=1
            continue
        if len(rescore)>1 and rescore[1][0]==relation[index]:
            top2+=1
            top3+=1
            top4+=1
            top5+=1
            seere+=1
            continue
        if len(rescore)>2 and rescore[2][0]==relation[index]:
            top3+=1
            top4+=1
            top5+=1
            seere+=1
            continue
        if len(rescore)>3 and rescore[3][0]==relation[index]:
            top4+=1
            top5+=1
            seere+=1
            continue
        if len(rescore)>4 and rescore[4][0]==relation[index]:
            top5+=1
            seere+=1
            continue
    else:
        if len(rescore)>0 and rescore[0][0]==relation[index]:
            top1+=1
            top2+=1
            top3+=1
            top4+=1
            top5+=1
            unseeac+=1
            unseere+=1
            continue
        if len(rescore)>1 and rescore[1][0]==relation[index]:
            top2+=1
            top3+=1
            top4+=1
            top5+=1
            unseere+=1
            continue
        if len(rescore)>2 and rescore[2][0]==relation[index]:
            top3+=1
            top4+=1
            top5+=1
            unseere+=1
            continue
        if len(rescore)>3 and rescore[3][0]==relation[index]:
            top4+=1
            top5+=1
            unseere+=1
            continue
        if len(rescore)>4 and rescore[4][0]==relation[index]:
            top5+=1
            unseere+=1
            continue

'''
print(" flase segement ")

with open(SCALE+'/'+MODEL+'/result.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        line=line.split(' %%%% ')
        ques=line[0].strip()
        #��������ж��룬���������ʵ�����ӽ��Ϊ�գ������Ҫ������ЩΪ�յ�����
        while ques!=question[index]:
            index+=1
        if ques==question[index] and len(line)>1:
            for i in range(1,len(line)):
                if line[i].split('\t')[1].strip()==relation[index]:
                    if i==1:
                        top1+=1
                        top2+=1
                        top3+=1
                        top4+=1
                        top5+=1
                        break
                    if i==2:
                        top2+=1
                        top3+=1
                        top4+=1
                        top5+=1
                        break
                    if i==3:
                        top3+=1
                        top4+=1
                        top5+=1
                        break
                    if i==4:
                        top4+=1
                        top5+=1
                        break
                    if i==5:
                        top5+=1
                        break
                    if i>5:
                        break
        line=f.readline()
'''


my_logger.warning(('���Լ�������',len(truth)))
my_logger.warning(('top1������',top1))
my_logger.warning(('top1�ʣ�',100*top1/len(truth)))
my_logger.warning(('top2������',top2))
my_logger.warning(('top2�ʣ�',100*top2/len(truth)))
my_logger.warning(('top3������',top3))
my_logger.warning(('top3�ʣ�',100*top3/len(truth)))
my_logger.warning(('top4������',top4))
my_logger.warning(('top4�ʣ�',100*top4/len(truth)))
my_logger.warning(('top5������',top5))
my_logger.warning(('top5�ʣ�',100*top5/len(truth)))
my_logger.warning(('������ϵ׼ȷ�ʣ�',seeac/len(quessee)))
my_logger.warning(('δ������ϵ׼ȷ�ʣ�',unseeac/len(quesunsee)))
my_logger.warning(('������ϵtop5�ʣ�',seere/len(quessee)))
my_logger.warning(('δ������ϵtop5�ʣ�',unseere/len(quesunsee)))

