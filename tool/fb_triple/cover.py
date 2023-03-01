#!/usr/bin/python3
# coding=gbk
from copy import deepcopy
from util import www2fb, processed_text, clean_uri
import logging
'''
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

triple=set()
headre=set()
with open('fb_triple.txt','r',encoding='utf-8') as f:
    lines=f.readlines()
    for i in lines:
        line=i.strip().split('\t')
        triple.add(tuple(line))
        headre.add(tuple([line[0],line[1]]))

num1=0
triple1=deepcopy(triple)
headre1=deepcopy(headre)
count1=set()
with open('all.txt','r',encoding='utf-8') as f:
    lines=f.readlines()
    for i in lines:
        line=i.strip().split('\t')
        triple1.add(tuple([line[1][3:],line[3][3:],line[4][3:]]))
        headre1.add(tuple([line[1][3:],line[3][3:]]))
        count1.add(tuple([line[1][3:],line[3][3:],line[4][3:]]))
num1=len(count1)

num2=0
triple2=deepcopy(triple)
headre2=deepcopy(headre)
count2=set()
with open('freebase-FB2M.txt','r',encoding='utf-8') as f:
    lines=f.readlines()
    for i in lines:
        items=i.strip().split('\t')
        entity1 = www2fb(items[0])
        entity2=www2fb(items[2]).split()[0]
        rel=www2fb(items[1])
        triple2.add(tuple([entity1,rel,entity2]))
        headre2.add(tuple([entity1,rel]))
        count2.add(tuple([entity1,rel,entity2]))
num2=len(count2)

add_tri=len(triple1)-len(triple)
add_hr=len(headre1)-len(headre)

add_tri=len(triple2)-len(triple)
add_hr=len(headre2)-len(headre)
'''
triple=set()
with open('fb_triple.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        triple.add(line)
        line=f.readline()

FLAG=True
with open('2Mtriple.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        num1=len(triple)
        triple.add(line)
        num2=len(triple)
        if num1!=num2:
            print(line)
            FLAG=False
            break
        line=f.readline()

if FLAG:
    print('cover')
else:
    print('not cover')
    