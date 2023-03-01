#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import pandas as pd

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

index=0
top1=0
top2=0
top3=0
top4=0
top5=0
num=0

entity_correct_top1 = 0
relationship_correct_top1 = 0

#f1=open('entity_miss.txt','w',encoding='utf-8')
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
            '''
            print('Ԥ��')
            print(line[1].split('\t')[0].strip())
            print(line[1].split('\t')[1].strip())
            print('��')
            print(entity_mid[index])
            print(relation[index])
            '''
            for i in range(1,len(line)):
                if SCALE=='small':
                    entity=entity_mid[index]
                else:
                    entity=entity_mid[index][3:]

                # mdh 2022-9-21
                if i==1 and (line[i].split('\t')[0].strip()==entity):
                    entity_correct_top1 += 1
                if i==1 and (line[i].split('\t')[1].strip()==relation[index]):
                    relationship_correct_top1 += 1

                if (line[i].split('\t')[0].strip()==entity) and (line[i].split('\t')[1].strip()==relation[index]):
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

        line=f.readline()
        index+=1

my_logger.warning('���Լ�������{}'.format(len(truth)))
my_logger.warning('top1������{}'.format(top1))
my_logger.warning('top1�ʣ�{}'.format(100*top1/len(truth)))
my_logger.warning('top2������{}'.format(top2))
my_logger.warning('top2�ʣ�{}'.format(top2/len(truth)))
my_logger.warning('top3������{}'.format(top3))
my_logger.warning('top3�ʣ�{}'.format(100*top3/len(truth)))
my_logger.warning('top4������{}'.format(top4))
my_logger.warning('top4�ʣ�{}'.format(top4/len(truth)))
my_logger.warning('top5������{}'.format(top5))
my_logger.warning('top5�ʣ�{}'.format(top5/len(truth)))

my_logger.warning("*****************")
my_logger.warning("")
my_logger.warning(" entity correct at top 1:         {}".format(entity_correct_top1 / len(truth) * 100.0))
my_logger.warning(
        " relationship correct at top 1:   {}".format(relationship_correct_top1 / len(truth) * 100.0))
#f1.close()
