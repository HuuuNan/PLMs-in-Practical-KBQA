#!/usr/local/bin/python
# -*- coding: gbk -*-
import pickle
from query_interface import query_en_relation
import logging


redict=pickle.load(open('redict.pkl','rb'))
origin=pickle.load(open('reachability_2M.pkl','rb'))

for i in origin.items():
    if i[0] not in redict.keys():
        redict[i[0]]=i[1]

pickle.dump(redict,open('redict1.pkl','wb'))
    