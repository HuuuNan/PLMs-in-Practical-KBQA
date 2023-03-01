#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import json
import pickle
import time
import os
import sys
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


topk=50
data=json.load(open(SCALE+'/'+MODEL+'/result.json','r',encoding='utf-8'))
f=open(SCALE+'/'+MODEL+'/result.txt','w',encoding='utf-8')
pred_num=0
sorttime=0

if SCALE=='small':
    de_dict=pickle.load(open("../indexes/degrees_2M.pkl",'rb'))
if SCALE=='large':
    de_dict=pickle.load(open("../indexes/degrees.pkl",'rb'))
if SCALE=='medium1':
    de_dict=pickle.load(open("../kb_105M/degrees.pkl",'rb'))
if SCALE=='medium2':
    de_dict=pickle.load(open("../kb_202M/degrees.pkl",'rb'))

for ques,ques1,paths,scores,re_scores in zip(data['questions'],data['origin_questions'],data['paths'],data['scores'],data['re_scores']):
    start=time.time()
    ques_result=[]
    #i�ļ�Ϊʵ�壬ֵΪ��ϵ·��
    for index,i in enumerate(paths.items()):
       res=re_scores[i[0]]
       for j in res:
           if i[0] in de_dict.keys():
               de_score=de_dict[i[0]][0]
           else:
               de_score=0
           ques_result.append([j[1],de_score,i[0],j[0]])
    ques_result.sort(key=lambda t: (t[0],t[1]), reverse=True)
    end=time.time()
    sorttime=sorttime+end-start
    f.write(ques1)
    if len(ques_result)<topk:
        num=len(ques_result)
    else:
        num=topk
    for k in range(0,num):
        f.write(' %%%% ')
        f.write(ques_result[k][2])
        f.write('\t')
        f.write(ques_result[k][3])
        f.write('\t')
        f.write(str(ques_result[k][0]))
        f.write('\t')
        f.write(str(ques_result[k][1]))
    f.write('\n')
    #��ʾԤ�����
    pred_num+=1
    if pred_num%500==0:
        print('��Ԥ�⣺',pred_num,'/',len(data['questions']))
        print('Ԥ����ȣ�',pred_num/len(data['questions']))

f.close()

my_logger.warning('����ʱ�䣺{}'.format(sorttime))

