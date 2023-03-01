import pandas as pd
import os
import math
import pickle

from argparse import ArgumentParser
from collections import defaultdict
import time
import pandas as pd
from torchtext import data
import sys
sys.path.append("..")
from mylogger import mylog

# set logger
my_logger = mylog.log_creater('./log', 'data_process_re-out_mdh')


def load_index(filename):
    my_logger.warn("Loading index map from {}".format(filename))
    with open(filename, 'rb') as handler:
        index = pickle.load(handler)
    return index

# Load predicted MIDs and relations for each question in valid/test set
def get_mids(filename, hits):
    my_logger.warn("Entity Source : {}".format(filename))
    id2mids = defaultdict(list)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split(' %%%% ')
        lineid = items[0]
        cand_mids = items[1:][:hits]
        for mid_entry in cand_mids:
            mid, mid_name, mid_type, score = mid_entry.split('\t')
            id2mids[lineid].append((mid, mid_name, mid_type, float(score)))
    return id2mids

def get_rels(filename, hits):
    my_logger.warn("Relation Source : {}".format(filename))
    id2rels = defaultdict(list)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split(' %%%% ')
        lineid = items[0].strip()
        rel = www2fb(items[1].strip())
        label = items[2].strip()
        score = items[3].strip()
        if len(id2rels[lineid]) < hits:
            id2rels[lineid].append((rel, label, float(score)))
    return id2rels


def get_questions(filename):
    my_logger.warn("getting questions ...")
    id2questions = {}
    id2goldmids = {}
    fin =open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        lineid = items[0].strip()
        mid = items[1].strip()
        question = items[5].strip()
        rel = items[3].strip()
        id2questions[lineid] = (question, rel)
        id2goldmids[lineid] = mid
    return id2questions, id2goldmids

def get_mid2wiki(filename):
    my_logger.warn("Loading Wiki")
    mid2wiki = defaultdict(bool)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        sub = rdf2fb(clean_uri(items[0]))
        mid2wiki[sub] = True
    return mid2wiki
'''
index_reach = load_index('../indexes/reachability_2M.pkl')
exist=[]
for i in index_reach.values():
    for j in i:
        if j not in exist:
            exist.append(j)

with open('./relation.txt','w',encoding='utf-8') as f:
    for i in exist:
        i=i.replace('.',' ')
        i=i.replace('_',' ')
        i=i[3:len(i)]
        f.write(i)
        f.write('\n')
'''
train=pd.read_table('../mydata/train.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
valid=pd.read_table('../mydata/valid.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
test=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])

re_list=train['relation']
with open('./train_re.txt','w',encoding='utf-8') as f1:
    for i in range(0,len(re_list)):
        re=re_list[i][3:len(re_list)].replace('.',' ')
        re=re.replace('_',' ')
        f1.write(re)
        f1.write('\n')
        
re_list=valid['relation']
with open('./valid_re.txt','w',encoding='utf-8') as f1:
    for i in range(0,len(re_list)):
        re=re_list[i][3:len(re_list)].replace('.',' ')
        re=re.replace('_',' ')
        f1.write(re)
        f1.write('\n')
        
re_list=test['relation']
with open('./test_re.txt','w',encoding='utf-8') as f1:
    for i in range(0,len(re_list)):
        re=re_list[i][3:len(re_list)].replace('.',' ')
        re=re.replace('_',' ')
        f1.write(re)
        f1.write('\n')