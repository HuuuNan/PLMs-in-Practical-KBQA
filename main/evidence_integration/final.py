# coding=utf-8
import os
import math
import pickle
import pandas as pd
import time

from argparse import ArgumentParser
from collections import defaultdict
from util import clean_uri, processed_text, www2fb, rdf2fb

import mylog
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

def evidence_integration(data_path, ent_path, rel_path, output_dir, index_reach, index_degrees, is_heuristics, HITS_ENT, HITS_REL):
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
            
    my_logger.warning("")
    my_logger.warning("found:              {}".format(found / len(id2goldmids) * 100.0))
    my_logger.warning("retrieved at top 1: {}".format(retrieved_top1 / len(id2goldmids) * 100.0))
    my_logger.warning("retrieved at top 2: {}".format(retrieved_top2 / len(id2goldmids) * 100.0))
    my_logger.warning("retrieved at top 3: {}".format(retrieved_top3 / len(id2goldmids) * 100.0))

    my_logger.warning("")
    my_logger.warning(" entity correct at top 1:         {}".format(entity_correct_top1 / len(id2goldmids) * 100.0))
    my_logger.warning(" relationship correct at top 1:   {}".format(relationship_correct_top1 / len(id2goldmids) * 100.0))
    return id2answers

if __name__=="__main__":
    TYPE = 'test'
    MODEL = 'bert-base-uncased'
    SCALE = 'small'  # small, medium1(105M), medium2(202M), large
    DISAMB = False # flase for direct entity linking result, true for entity linking result after entity disamb

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

    if DISAMB:
        parser.add_argument('--ent_path', type=str, default="../entity_disamb/link_result/"+SCALE+"/"+MODEL+"/"+TYPE+"-h100.txt", help='path to the entity linking results')
    else:
        parser.add_argument('--ent_path', type=str,default="../entity_linking/" + SCALE + "/" + MODEL + "/" + TYPE + "-h100.txt", help='path to the entity linking results')

    parser.add_argument('--rel_path', type=str, default="../relation_prediction/"+SCALE+'/'+MODEL+'_results/'+TYPE+".txt", help='path to the relation prediction results')
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
    time2=time.time()
    process_time=time2-time1
    my_logger.warning('process time:{}'.format(process_time))
    
    time3=time.time()
    test_answers = evidence_integration(args.data_path, args.ent_path, args.rel_path, output_dir, index_reach, index_degrees, args.heuristics, args.hits_ent, args.hits_rel)
    time4=time.time()
    finalstep_time=time4-time3
    if TYPE=='valid':
        my_logger.warning('valid time:{}'.format(finalstep_time))
    if TYPE=='test':
        my_logger.warning('test time:{}'.format(finalstep_time))
