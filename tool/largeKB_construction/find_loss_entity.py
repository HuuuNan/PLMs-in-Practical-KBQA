#!/usr/local/bin/python
# -*- coding: gbk -*-
'''
import pickle
re_dict=pickle.load(open('redict.pkl','rb'))
print(len(re_dict))
'''
entity=[]
with open('entity.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        entity.append(line)
        line=f.readline()
print(len(entity))
num=0
miss=[]
with open('all.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        num+=1
        if num%1000==0:
            print(num)
        line=line.strip().split()[1][3:]
        if line not in entity:
            miss.append(line)
            print(line)
        line=f.readline()
        
with open('miss_entity1.txt','w',encoding='utf-8') as f:
    for i in miss:
        f.write(i)
        f.write('\n')
