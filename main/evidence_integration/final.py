# coding=utf-8
import os
import math
import pickle
import pandas as pd
import time

from argparse import ArgumentParser
from collections import defaultdict
from util import clean_uri, processed_text, www2fb, rdf2fb

import sys
sys.path.append("..")
from mylogger import mylog
file_name = os.path.basename(__file__)
file_name = file_name[0:len(file_name)-3]


# Load up reachability graph

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
            mid, mid_name,score = mid_entry.split('\t')
            id2mids[lineid].append((mid, mid_name,float(score)))
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
    my_logger.warning("Loading Wiki")
    mid2wiki = defaultdict(bool)
    fin = open(filename)
    for line in fin.readlines():
        items = line.strip().split('\t')
        sub = rdf2fb(clean_uri(items[0]))
        mid2wiki[sub] = True
    return mid2wiki

def evidence_integration(data_path, ent_path, rel_path, output_dir, index_reach, index_degrees, mid2wiki, is_heuristics, HITS_ENT, HITS_REL):
    id2questions, id2goldmids = get_questions(data_path)
    id2mids = get_mids(ent_path, HITS_ENT)
    id2rels = get_rels(rel_path, HITS_REL)
    file_base_name = os.path.basename(data_path)
    #fout = open(os.path.join(output_dir, file_base_name), 'w')

    id2answers = defaultdict(list)
    found, notfound_both, notfound_mid, notfound_rel = 0, 0, 0, 0
    retrieved, retrieved_top1, retrieved_top2, retrieved_top3 = 0, 0, 0, 0
    lineids_found1 = []
    lineids_found2 = []
    lineids_found3 = []
    entity_correct_top1, entity_correct_top2, entity_correct_top3 = 0, 0, 0
    relationship_correct_top1, relationship_correct_top2, relationship_correct_top3 = 0, 0, 0

 
    # for every lineid
    for line_id in id2goldmids:
        if line_id not in id2mids and line_id not in id2rels:
            notfound_both += 1
            continue
        elif line_id not in id2mids:
            notfound_mid += 1
            continue
        elif line_id not in id2rels:
            notfound_rel += 1
            continue

        found += 1
        question, truth_rel = id2questions[line_id]
        #truth_rel = www2fb(truth_rel)
        #truth_mid = id2goldmids[line_id]
        
        if SCALE=='small':
            truth_mid = id2goldmids[line_id]
        else:
            truth_mid = id2goldmids[line_id][3:]
            truth_rel=truth_rel[3:]
        
        mids = id2mids[line_id]
        rels = id2rels[line_id]
        
        if is_heuristics:
            for (mid, mid_name,mid_score) in mids:
                for (rel, rel_label, rel_log_score) in rels:
                    # if this (mid, rel) exists in FB
                    #if mid in index_reach.keys() and rel in index_reach[mid]:
                    if mid in index_reach.keys() and rel in index_reach[mid]:
                        rel_score = math.exp(float(rel_log_score))
                        # comb_score = (float(mid_score)**1) + (rel_score*1000)
                        comb_score =(rel_score*1000)
                        #id2answers[line_id].append((mid, rel, mid_name, mid_type, mid_score, rel_score, comb_score, int(mid2wiki[mid]), int(index_degrees[mid][0])))
                        if mid in index_degrees.keys():
                            de_score=index_degrees[mid][0]
                        else:
                            de_score=0
                        id2answers[line_id].append((mid,rel,comb_score,de_score))
                    # I cannot use retrieved here because I use contain different name_type
                    # if mid ==truth_mid and rel == truth_rel:
                    #     retrieved += 1
            id2answers[line_id].sort(key=lambda t: (t[2],[3]), reverse=True)
        else:
            id2answers[line_id] = [(mids[0][0], rels[0][0])]
        '''
        # write to file
        fout.write("{}".format(line_id))
        for answer in id2answers[line_id]:
            mid, rel, mid_name, mid_type, mid_score, rel_score, comb_score, _, _ = answer
            fout.write(" %%%% {}\t{}\t{}\t{}\t{}\t{}".format(mid, rel, mid_name, mid_score, rel_score, comb_score))
        fout.write('\n')
        '''
        if len(id2answers[line_id]) >= 1 and id2answers[line_id][0][0] == truth_mid \
                and id2answers[line_id][0][1] == truth_rel:
            retrieved_top1 += 1
            retrieved_top2 += 1
            retrieved_top3 += 1
            lineids_found1.append(line_id)
        elif len(id2answers[line_id]) >= 2 and id2answers[line_id][1][0] == truth_mid \
                and id2answers[line_id][1][1] == truth_rel:
            retrieved_top2 += 1
            retrieved_top3 += 1
            lineids_found2.append(line_id)
        elif len(id2answers[line_id]) >= 3 and id2answers[line_id][2][0] == truth_mid \
                and id2answers[line_id][2][1] == truth_rel:
            retrieved_top3 += 1
            lineids_found3.append(line_id)

        if len(id2answers[line_id]) >= 1 and id2answers[line_id][0][0] == truth_mid:
            entity_correct_top1 += 1
        elif len(id2answers[line_id]) >= 2 and id2answers[line_id][1][0] == truth_mid:
            entity_correct_top2 += 1
        elif len(id2answers[line_id]) >= 3 and id2answers[line_id][2][0] == truth_mid:
            entity_correct_top3 += 1

        if len(id2answers[line_id]) >= 1 and id2answers[line_id][0][1] == truth_rel:
            relationship_correct_top1 += 1
        elif len(id2answers[line_id]) >= 2 and id2answers[line_id][1][1] == truth_rel:
            relationship_correct_top2 += 1
        elif len(id2answers[line_id]) >= 3 and id2answers[line_id][2][1] == truth_rel:
            relationship_correct_top3 += 1
    '''
    total=pd.read_table('../mydata/'+TYPE+'.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])  
    top1=pd.DataFrame(columns=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"]) 
    top2=pd.DataFrame(columns=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"]) 
    top3=pd.DataFrame(columns=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"]) 
    rest=pd.DataFrame(columns=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
    num1=0
    num2=0
    num3=0
    num4=0
    for index, row in total.iterrows():
        FLAG=False
        if row['lineid'] in lineids_found1:
            top1.loc[num1]=row
            num1+=1
            FLAG=True
        if row['lineid'] in lineids_found2:
            top2.loc[num2]=row
            num2+=1
            FLAG=True
        if row['lineid'] in lineids_found3:
            top3.loc[num3]=row
            num3+=1
            FLAG=True
        if not FLAG:
            rest.loc[num4]=row
            num4+=1
    top1.to_csv('./'+args.model+'_????????/'+TYPE+'/top1.txt',sep='\t',header=None,index=False)
    top2.to_csv('./'+args.model+'_????????/'+TYPE+'/top2.txt',sep='\t',header=None,index=False)
    top3.to_csv('./'+args.model+'_????????/'+TYPE+'/top3.txt',sep='\t',header=None,index=False)
    rest.to_csv('./'+args.model+'_????????/'+TYPE+'/rest.txt',sep='\t',header=None,index=False)
    '''
    my_logger.warning("")
    my_logger.warning("found:              {}".format(found / len(id2goldmids) * 100.0))
    my_logger.warning("retrieved at top 1: {}".format(retrieved_top1 / len(id2goldmids) * 100.0))
    my_logger.warning("retrieved at top 2: {}".format(retrieved_top2 / len(id2goldmids) * 100.0))
    my_logger.warning("retrieved at top 3: {}".format(retrieved_top3 / len(id2goldmids) * 100.0))

    my_logger.warning("")
    my_logger.warning(" entity correct at top 1:         {}".format(entity_correct_top1 / len(id2goldmids) * 100.0))
    my_logger.warning(" relationship correct at top 1:   {}".format(relationship_correct_top1 / len(id2goldmids) * 100.0))
    #print("retrieved at inf:   {}".format(retrieved / len(id2goldmids) * 100.0))
    #fout.close()
    return id2answers

if __name__=="__main__":
    TYPE = 'valid'
    MODEL = 'albert-base-v2'
    SCALE = 'small'  # small, medium1(105M), medium2(202M), large

    # set logger
    global my_logger
    my_logger = mylog.log_creater('./log', file_name + '_' + MODEL + '_' + SCALE + '_' + TYPE + '-out')
    my_logger.warning("\n new process start --\n")
    my_logger.warning(MODEL + '-' + SCALE + '-' + TYPE)

    parser = ArgumentParser(description='Perform evidence integration')
    parser.add_argument('--model', type=str, default=MODEL)
    if SCALE=='small':
        parser.add_argument('--index_reachpath', type=str, default="../indexes/reachability_2M.pkl",help='path to the pickle for the reachability index')
    if SCALE=='large':
        parser.add_argument('--index_reachpath', type=str, default="../indexes/redict.pkl",
                        help='path to the pickle for the reachability index')
    if SCALE=='medium1':
        parser.add_argument('--index_reachpath', type=str, default="../kb_105M/redict.pkl",
                        help='path to the pickle for the reachability index')
    if SCALE=='medium2':
        parser.add_argument('--index_reachpath', type=str, default="../kb_202M/redict.pkl",
                        help='path to the pickle for the reachability index')
    if SCALE=='small':
        parser.add_argument('--index_degreespath', type=str, default="../indexes/degrees_2M.pkl",
                        help='path to the pickle for the index with the degree counts')
    if SCALE=='large':
        parser.add_argument('--index_degreespath', type=str, default="../indexes/degrees.pkl",
                        help='path to the pickle for the index with the degree counts')
    if SCALE=='medium1':
        parser.add_argument('--index_degreespath', type=str, default="../kb_105M/degrees.pkl",
                        help='path to the pickle for the index with the degree counts')
    if SCALE=='medium2':
        parser.add_argument('--index_degreespath', type=str, default="../kb_202M/degrees.pkl",
                        help='path to the pickle for the index with the degree counts')
    parser.add_argument('--data_path', type=str, default="../mydata/"+TYPE+".txt")

    # modified by mdh
    step_num = 2
    my_logger.warning("step {} result :".format(step_num))

    if(step_num == 2):
        parser.add_argument('--ent_path', type=str, default="../entity_disamb/link_result/"+SCALE+"/"+MODEL+"/"+TYPE+"-h100.txt", help='path to the entity linking results')
    else:
        parser.add_argument('--ent_path', type=str,default="../entity_linking/" + SCALE + "/" + MODEL + "/" + TYPE + "-h100.txt", help='path to the entity linking results')

    # parser.add_argument('--ent_path', type=str,
    #                     default="../entity_disamb/link_result/" + SCALE + "/" + MODEL + "/" + TYPE + "-h100.txt",
    #                     help='path to the entity linking results')
    parser.add_argument('--rel_path', type=str, default="../rel_prediction/"+SCALE+'/'+MODEL+'_results/'+TYPE+".txt", help='path to the relation prediction results')
    parser.add_argument('--wiki_path', type=str, default="../data/fb2w.nt")
    parser.add_argument('--hits_ent', type=int, default=50, help='the hits here has to be <= the hits in entity linking')
    parser.add_argument('--hits_rel', type=int, default=5, help='the hits here has to be <= the hits in relation prediction retrieval')
    parser.add_argument('--no_heuristics', action='store_false', help='do not use heuristics', dest='heuristics')
    parser.add_argument('--output_dir', type=str, default=SCALE)
    args = parser.parse_args()
    my_logger.warning(args)
    
    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    time1=time.time()
    index_reach = load_index(args.index_reachpath)
    index_degrees = load_index(args.index_degreespath)
    mid2wiki = get_mid2wiki(args.wiki_path)
    time2=time.time()
    process_time=time2-time1
    my_logger.warning('??????????????{}'.format(process_time))
    
    time3=time.time()
    test_answers = evidence_integration(args.data_path, args.ent_path, args.rel_path, output_dir, index_reach, index_degrees, mid2wiki, args.heuristics, args.hits_ent, args.hits_rel)
    time4=time.time()
    finalstep_time=time4-time3
    if TYPE=='valid':
        my_logger.warning('????????????????{}'.format(finalstep_time))
    if TYPE=='test':
        my_logger.warning('????????????????{}'.format(finalstep_time))