# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 13:21:32 2021

@author: Lenovo
"""

import pandas as pd
from sklearn.utils import shuffle

train=pd.read_table('../mydata/train.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])    
with open('./train.txt','w',encoding='utf-8') as f:
    for i in range(0,len(train)):
        x=train['question'][i].split(' ')
        y=train['tags'][i].split(' ')
        if ',' not in y[0]:
            for j in range(0,len(x)):
                f.write(x[j])
                f.write(' ')
                f.write(y[j])
                f.write('\n')
            f.write('\n')
        else:
            for j in range(0,len(x)):
                f.write(x[j])
                f.write(' ')
                if 'O' in y[j]:
                    f.write('O')
                else:
                    f.write('I')
                f.write('\n')
            f.write('\n')
     
valids=[]       
valid=pd.read_table('../mydata/valid.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])    
with open('./valid.txt','w',encoding='utf-8') as f:
    for i in range(0,len(valid)):
        valids.append(valid['question'][i])
        x=valid['question'][i].split(' ')
        y=valid['tags'][i].split(' ')
        if ',' not in y[0]:
            for j in range(0,len(x)):
                f.write(x[j])
                f.write(' ')
                f.write(y[j])
                f.write('\n')
            f.write('\n')
        else:
            for j in range(0,len(x)):
                f.write(x[j])
                f.write(' ')
                if 'O' in y[j]:
                    f.write('O')
                else:
                    f.write('I')
                f.write('\n')
            f.write('\n')
    
tests=[]
test=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])    
with open('./test.txt','w',encoding='utf-8') as f:
    for i in range(0,len(test)):
        tests.append(test['question'][i])
        x=test['question'][i].split(' ')
        y=test['tags'][i].split(' ')
        if ',' not in y[0]:
            for j in range(0,len(x)):
                f.write(x[j])
                f.write(' ')
                f.write(y[j])
                f.write('\n')
            f.write('\n')
        else:
            for j in range(0,len(x)):
                f.write(x[j])
                f.write(' ')
                if 'O' in y[j]:
                    f.write('O')
                else:
                    f.write('I')
                f.write('\n')
            f.write('\n')
            
with open('./valid_sentence.txt','w',encoding='utf-8') as f:
    for i in valids:
        f.write(i)
        f.write('\n')
        
with open('./test_sentence.txt','w',encoding='utf-8') as f:
    for i in tests:
        f.write(i)
        f.write('\n')
            
