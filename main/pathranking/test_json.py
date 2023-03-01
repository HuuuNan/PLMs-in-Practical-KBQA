import random
import json
import pandas as pd
import os
import math
import pickle
import time
from argparse import ArgumentParser
from collections import defaultdict
import time
import pandas as pd
import sys

sys.path.append("..")
from mylogger import mylog

file_name = os.path.basename(__file__)
file_name = file_name[0:len(file_name) - 3]

MODEL = 'distilroberta-base'
SCALE = 'large'

# set logger
my_logger = mylog.log_creater('./log', file_name + '_' + MODEL + '_' + SCALE + '-out')
my_logger.warning("\n new process start  \n")
my_logger.warning(MODEL + '-' + SCALE)

os.makedirs(SCALE + '/' + MODEL, exist_ok=True)
test = pd.read_table('../mydata/test.txt', header=None,
                     names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
test_dict = json.load(open(SCALE + '/' + MODEL + '/test_dict.json', 'r', encoding='utf-8'))


def load_index(filename):
    my_logger.warning("Loading index map from {}".format(filename))
    with open(filename, 'rb') as handler:
        index = pickle.load(handler)
    return index


# Load predicted MIDs and relations for each question in valid/test set
def get_mids(filename, hits):
    my_logger.warning("Entity Source : {}".format(filename))
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
    my_logger.warning("Relation Source : {}".format(filename))
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
    my_logger.warning("getting questions ...")
    id2questions = {}
    id2goldmids = {}
    fin = open(filename)
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
    my_logger.warning("Loading Wiki")
    mid2wiki = defaultdict(bool)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        sub = rdf2fb(clean_uri(items[0]))
        mid2wiki[sub] = True
    return mid2wiki


if SCALE == 'small':
    index_reach = load_index('../indexes/reachability_2M.pkl')
if SCALE == 'large':
    index_reach = load_index('../indexes/redict.pkl')
if SCALE == 'medium1':
    index_reach = load_index('../kb_105M/redict.pkl')
if SCALE == 'medium2':
    index_reach = load_index('../kb_202M/redict.pkl')

entity_num = 50
'''
total_question=[]
total_path=[]
total_score=[]
total_entity=[]
origin_question=[]
with open('./entity_linking/test-h100.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        line=line.split('%%%%')
        num=0
        path=[]
        score=[]
        entity=[]
        for i in line[1:len(line)]:
            rel_list=[]
            if i.split('\t')[0].strip() in index_reach:
                num+=1
                for j in index_reach[i.split('\t')[0].strip()]:
                    j=j.replace('.',' ')
                    j=j.replace('_',' ')
                    j=j[3:len(j)]
                    rel_list.append(j) 
                path.extend(rel_list)
                for k in range(0,len(rel_list)): 
                    score.append(i.split('\t')[-1].strip())
                    entity.append(i.split('\t')[0].strip())
                if num==entity_num:
                    break
        if len(path)!=0:
            total_question.append(test_dict[line[0].strip()]['question'])
            total_path.append(path)
            total_score.append(score)
            total_entity.append(entity)
            origin_question.append(test_dict[line[0].strip()]['origin_question'])
        line=f.readline()

test_json=dict()
test_json["questions"]=total_question
test_json["paths"]=total_path
test_json['scores']=total_score
test_json['entities']=total_entity
test_json['origin_questions']=origin_question
with open('./data/test.json', 'w') as f:
    json.dump(test_json, f,ensure_ascii=False)
'''
total_question = []
total_path = []
total_score = []
origin_question = []
relation = []
total_relation = []
start = time.time()

step_num = 1
entity_sources_path = ""
if step_num == 1:
    entity_sources_path = "../entity_linking/" + SCALE + "/" + MODEL + "/" + "test-h100.txt"
else:
    entity_sources_path = '../entity_disamb/link_result/' + SCALE + '/' + MODEL + '/test-h100.txt'

with open(entity_sources_path, 'r', encoding='utf-8') as f:
    line = f.readline()
    while line != '':
        line = line.strip()
        line = line.split('%%%%')
        num = 0
        path = dict()
        score = []
        rel_all = []
        for i in line[1:len(line)]:
            rel_list = []
            if i.split('\t')[0].strip() in index_reach:
                num += 1
                for j in index_reach[i.split('\t')[0].strip()]:
                    j = j.replace('.', ' ')
                    j = j.replace('_', ' ')
                    if SCALE == 'small':
                        j = j[3:len(j)]
                    if j != 'type object name' and j != 'common topic alias':
                        rel_list.append(j)
                        if j not in rel_all:
                            rel_all.append(j)
                path[i.split('\t')[0].strip()] = rel_list
                score.append(i.split('\t')[-1].strip())
                if num == entity_num:
                    break
        if len(path) != 0:
            total_question.append(test_dict[line[0].strip()]['question'])
            total_path.append(path)
            total_score.append(score)
            origin_question.append(test_dict[line[0].strip()]['origin_question'])
            total_relation.append(rel_all)
        line = f.readline()
end = time.time()

test_json = dict()
test_json["questions"] = total_question
test_json["paths"] = total_path
test_json['scores'] = total_score
test_json['origin_questions'] = origin_question
test_json['relations'] = total_relation
with open(SCALE + '/' + MODEL + '/test.json', 'w') as f:
    json.dump(test_json, f, ensure_ascii=False)
my_logger.warning('测试集关系查询时间：{}'.format(end - start))

my_logger.warning("\n  Program End \n")
