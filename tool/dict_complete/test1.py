#!/usr/bin/python3
# coding=gbk
import pickle

a=pickle.load(open('reachability_2M.pkl','rb'))

b=pickle.load(open('redict.pkl','rb'))

def convert(a):
    temp=set()
    for i in a:
        temp.add(i[3:])
    return temp

FLAG=True
for i in a.items():
    c=convert(i[1])
    if (b.get(i[0][3:]) is None) or (c & set(b[i[0][3:]])!=c):
        FLAG=False
        print(b.get(i[0][3:]))
        print(i[1])
        break
if FLAG:
    print('cover')
else:
    print('not cover')

