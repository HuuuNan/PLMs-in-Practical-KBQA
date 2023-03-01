#!/usr/bin/python3
# coding=gbk
import pickle
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

old=pickle.load(open('reachability_2M.pkl','rb'))
new=pickle.load(open('redict.pkl','rb'))

def convert(a):
    temp=set()
    for i in a:
        temp.append(i[3:])
    return temp

num=0
for i in old.items():
    num+=1
    if num%1000==0:
        logging.info(num)
    if new.get(i[0][3:]) is None:
        new[i[0][3:]]=convert(i[1])
    else:
        set1=set(new[i[0][3:]])
        set2=set(convert(i[1]))
        set3=set1|set2
        new[i[0][3:]]=set3
        
pickle.dump(new, open('redict1.pkl','wb'))

    