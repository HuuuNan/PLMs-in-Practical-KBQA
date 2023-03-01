import os
import pickle

from collections import defaultdict
from tqdm import tqdm
from argparse import ArgumentParser
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from util import clean_uri, processed_text, www2fb
import time

import sys
sys.path.append("..")
from mylogger import mylog
file_name = os.path.basename(__file__)
file_name = file_name[0:len(file_name)-3]

inverted_index = defaultdict(list)
stopword = set(stopwords.words('english'))

def get_ngram(text):
    #ngram = set()
    ngram = []
    tokens = text.split()
    for i in range(len(tokens)+1):
        for j in range(i):
            if i-j <= 3:
                #ngram.add(" ".join(tokens[j:i]))
                temp = " ".join(tokens[j:i])
                if temp not in ngram:
                    ngram.append(temp)
    #ngram = list(ngram)
    ngram = sorted(ngram, key=lambda x: len(x.split()), reverse=True)
    return ngram

def get_stat_inverted_index(filename):
    """
    Get the number of entry and max length of the entry (How many mid in an entry)
    """
    with open(filename, "rb") as handler:
        global  inverted_index
        inverted_index = pickle.load(handler)
        inverted_index = defaultdict(str, inverted_index)
    my_logger.warning("Total type of text: {}".format(len(inverted_index)))
    max_len = 0
    _entry = ""
    for entry, value in inverted_index.items():
        if len(value) > max_len:
            max_len = len(value)
            _entry = entry
    my_logger.warning("Max Length of entry is {}, text is {}".format(max_len, _entry))


def entity_linking(data_type, predictedfile, goldfile, HITS_TOP_ENTITIES, output):
    my_logger.warning("Source : {}".format(predictedfile))
    predicted = open(predictedfile)
    gold = open(goldfile)
    fout = open(output, 'w')
    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    top20 = 0
    top50 = 0
    top100 = 0

    for idx, (line, gold_id) in tqdm(enumerate(zip(predicted.readlines(), gold.readlines()))):
        total += 1
        line = line.strip().split(" %%%% ")
        if len(line)!=2:
            continue
        gold_id = gold_id.strip().split('\t')[1]
        # Use n-gram to filter most of the keys
        # We use the list to maintain the candidates
        # for counting
        # print(line[1])
        C = []
        # C_counts = []
        C_scored = []
        line_id = line[0]
        tokens = get_ngram(line[1])

        if len(tokens) > 0:
            maxlen = len(tokens[0].split())
        for item in tokens:
            if len(item.split()) < maxlen and len(C) == 0:
                maxlen = len(item.split())
            if len(item.split()) < maxlen and len(C) > 0:
                break
            if item in stopword:
                continue
            C.extend(inverted_index[item])
            # if len(C) > 0:
            #     break

        for mid_text_type in sorted(set(C)):
            score = fuzz.ratio(mid_text_type[1], line[1]) / 100.0
            # C_counts format : ((mid, text, type), score_based_on_fuzz)
            C_scored.append((mid_text_type, score))

        C_scored.sort(key=lambda t: t[1], reverse=True)
        cand_mids = C_scored[:HITS_TOP_ENTITIES]
        fout.write("{}".format(line_id))
        for mid_text_type, score in cand_mids:
            #fout.write(" %%%% {}\t{}\t{}".format(mid_text_type[0], mid_text_type[1], score))
            fout.write(" %%%% {}\t{}\t{}".format(mid_text_type[0], mid_text_type[1],score))
        fout.write('\n')
        gold_id = www2fb(gold_id)
        midList = [x[0][0] for x in cand_mids]
        #自己整理的entity mid前面没有fb:
        if SCALE!='small':
            gold_id=gold_id[3:]
        if gold_id in midList[:1]:
            top1 += 1
        if gold_id in midList[:3]:
            top3 += 1
        if gold_id in midList[:5]:
            top5 += 1
        if gold_id in midList[:10]:
            top10 += 1
        if gold_id in midList[:20]:
            top20 += 1
        if gold_id in midList[:50]:
            top50 += 1
        if gold_id in midList[:100]:
            top100 += 1
    
    my_logger.warning(data_type)
    my_logger.warning("Top1 Entity Linking Accuracy: {}".format(top1 / total))
    my_logger.warning("Top3 Entity Linking Accuracy: {}".format(top3 / total))
    my_logger.warning("Top10 Entity Linking Accuracy: {}".format(top10 / total))
    my_logger.warning("Top20 Entity Linking Accuracy: {}".format(top20 / total))
    my_logger.warning("Top50 Entity Linking Accuracy: {}".format(top50 / total))
    my_logger.warning("Top100 Entity Linking Accuracy: {}".format(top100 / total))
    
def entity_linking_train(goldfile, HITS_TOP_ENTITIES, output):
    gold = open(goldfile)
    fout = open(output, 'w')
    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    top20 = 0
    top50 = 0
    top100 = 0

    for idx, gold_id in tqdm(enumerate(gold.readlines())):
        total += 1
        line_id=gold_id.strip().split('\t')[0]
        mention=gold_id.strip().split('\t')[2]
        gold_id = gold_id.strip().split('\t')[1]
        # Use n-gram to filter most of the keys
        # We use the list to maintain the candidates
        # for counting
        # print(line[1])
        C = []
        # C_counts = []
        C_scored = []

        tokens = get_ngram(mention)

        if len(tokens) > 0:
            maxlen = len(tokens[0].split())
        for item in tokens:
            if len(item.split()) < maxlen and len(C) == 0:
                maxlen = len(item.split())
            if len(item.split()) < maxlen and len(C) > 0:
                break
            if item in stopword:
                continue
            C.extend(inverted_index[item])
            # if len(C) > 0:
            #     break

        for mid_text_type in sorted(set(C)):
            score = fuzz.ratio(mid_text_type[1], mention) / 100.0
            # C_counts format : ((mid, text, type), score_based_on_fuzz)
            C_scored.append((mid_text_type, score))

        C_scored.sort(key=lambda t: t[1], reverse=True)
        cand_mids = C_scored[:HITS_TOP_ENTITIES]
        fout.write("{}".format(line_id))
        for mid_text_type, score in cand_mids:
            #fout.write(" %%%% {}\t{}\t{}".format(mid_text_type[0], mid_text_type[1], score))
            fout.write(" %%%% {}\t{}\t{}".format(mid_text_type[0], mid_text_type[1],score))
        fout.write('\n')
        gold_id = www2fb(gold_id)
        midList = [x[0][0] for x in cand_mids]
        #自己整理的entity mid前面没有fb:
        if SCALE!='small':
            gold_id=gold_id[3:]
        if gold_id in midList[:1]:
            top1 += 1
        if gold_id in midList[:3]:
            top3 += 1
        if gold_id in midList[:5]:
            top5 += 1
        if gold_id in midList[:10]:
            top10 += 1
        if gold_id in midList[:20]:
            top20 += 1
        if gold_id in midList[:50]:
            top50 += 1
        if gold_id in midList[:100]:
            top100 += 1
    
    my_logger.warning("Top1 Entity Linking Accuracy: {}".format(top1 / total*100))
    my_logger.warning("Top3 Entity Linking Accuracy: {}".format(top3 / total*100))
    my_logger.warning("Top5 Entity Linking Accuracy: {}".format(top5 / total*100))
    my_logger.warning("Top10 Entity Linking Accuracy: {}".format(top10 / total*100))
    my_logger.warning("Top20 Entity Linking Accuracy: {}".format(top20 / total*100))
    my_logger.warning("Top50 Entity Linking Accuracy: {}".format(top50 / total*100))
    my_logger.warning("Top100 Entity Linking Accuracy: {}".format(top100 / total*100))


if __name__=="__main__":
    MODEL='bert-base-uncased'
    SCALE='small' #small, medium1(105M), medium2(202M), large

    # set logger
    my_logger = mylog.log_creater('./log', file_name + '_' + MODEL + '_' + SCALE + '-out')
    my_logger.warning("\n new process start  \n")
    my_logger.warning(MODEL + '-' + SCALE)

    parser = ArgumentParser(description='Perform entity linking')
    parser.add_argument('--model', type=str, default=MODEL)
    if SCALE=='small':
        parser.add_argument('--index_ent', type=str, default="../indexes/entity_2M.pkl",help='path to the pickle for the inverted entity index')
    if SCALE=='large':
        parser.add_argument('--index_ent', type=str, default="../indexes/entity.pkl",
                        help='path to the pickle for the inverted entity index')
    if SCALE=='medium1':
        parser.add_argument('--index_ent', type=str, default="../kb_105M/entity.pkl",
                        help='path to the pickle for the inverted entity index')
    if SCALE=='medium2':
        parser.add_argument('--index_ent', type=str, default="../kb_202M/entity.pkl",
                        help='path to the pickle for the inverted entity index')
    parser.add_argument('--data_dir', type=str, default="../mydata")
    parser.add_argument('--query_dir', type=str, default="../实体识别/"+MODEL+"_query")
    parser.add_argument('--hits', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default=SCALE)
    args = parser.parse_args()
    my_logger.warning(args)

    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)
     
    get_stat_inverted_index(args.index_ent)
    '''
    time0 = time.time()
    entity_linking_train(
                    os.path.join(args.data_dir, "train.txt"),
                    args.hits,
                    os.path.join(output_dir, "train-h{}.txt".format(args.hits)))
    
    time1=time.time()
    train_time = time1-time0
    '''
    entity_linking("valid",
                    os.path.join(args.query_dir, "query.valid"),
                    os.path.join(args.data_dir, "valid.txt"),
                    args.hits,
                    os.path.join(output_dir, "valid-h{}.txt".format(args.hits)))
    time2=time.time()
    valid_time=time2-time1
    '''
    time3=time.time()
    entity_linking("test",
                    os.path.join(args.query_dir, "query.test"),
                    os.path.join(args.data_dir, "test.txt"),
                    args.hits,
                    os.path.join(output_dir, "test-h{}.txt".format(args.hits)))
    time4=time.time()
    test_time=time4-time3
    '''
    my_logger.warning('模型训练时间：{}'.format(train_time))
    my_logger.warning('验证集实体链接时间：{}'.format(valid_time))
    my_logger.warning('测试集实体链接时间：{}'.format(test_time))

    my_logger.warning("\n  Program End \n")