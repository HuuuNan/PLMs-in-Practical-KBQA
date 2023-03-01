import json
import os

import sys

sys.path.append("..")
from mylogger import mylog

file_name = os.path.basename(__file__)
file_name = file_name[0:len(file_name) - 3]

MODEL = 'gpt2'
SCALE = 'large'

# set logger
my_logger = mylog.log_creater('./log', file_name + '_' + MODEL + '_' + SCALE + '-out')
my_logger.warning("\n new process start  \n")
my_logger.warning(MODEL + '-' + SCALE)

os.makedirs('link_result/' + SCALE + '/' + MODEL, exist_ok=True)


def convert(stage):
    result = json.load(open('result/' + SCALE + '/' + MODEL + '/' + stage + '.json', 'r', encoding='utf-8'))

    with open('data/' + stage + '.txt', 'r', encoding='utf-8') as f:
        gold = f.readlines()
    with open('../entity_linking/' + SCALE + '/' + MODEL + '/' + stage + '-h100.txt', 'r', encoding='utf-8') as f:
        text = f.readlines()

    total = len(gold)
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    top20 = 0
    top50 = 0
    top100 = 0

    f1 = open('link_result/' + SCALE + '/' + MODEL + '/' + stage + '-h100.txt', 'w', encoding='utf-8')
    num = 0
    for i in text:
        linelist = i.split(' %%%% ')
        lineid = linelist[0].split('-')[-1]
        newline = ''
        if result.get(lineid) is not None:
            namelist = []
            midlist = []
            for j in linelist[1:]:
                namelist.append(j.split('\t')[1])
                midlist.append(j.split('\t')[0])
            newline += linelist[0]
            tempdict = result[lineid]
            relist = []
            for index in range(0, len(midlist)):
                if SCALE == 'small':
                    relist.append([midlist[index], namelist[index], tempdict[midlist[index][3:]], (-1) * index])
                else:
                    relist.append([midlist[index], namelist[index], tempdict[midlist[index]], (-1) * index])
            relist.sort(key=lambda t: (t[2], [3]), reverse=True)
            for j in relist:
                newline = newline + ' %%%% ' + '{}\t{}\t{}'.format(j[0], j[1], j[2])
            newline += '\n'
        else:
            newline = i
        f1.write(newline)
        # evaluate matrix
        midList = []
        newlist = newline.split(' %%%% ')
        for j in newlist[1:]:
            midList.append(j.split('\t')[0])
        gold_id = gold[int(lineid) - 1].split('\t')[1]
        if SCALE != 'small':
            gold_id = gold_id[3:]

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

    my_logger.warning('********** {} result **********'.format(stage))
    my_logger.warning("Top1 Entity Linking Accuracy: {}".format(100 * top1 / total))
    my_logger.warning("Top3 Entity Linking Accuracy: {}".format(100 * top3 / total))
    my_logger.warning("Top5 Entity Linking Accuracy: {}".format(100 * top5 / total))
    my_logger.warning("Top10 Entity Linking Accuracy: {}".format(100 * top10 / total))
    my_logger.warning("Top20 Entity Linking Accuracy: {}".format(100 * top20 / total))
    my_logger.warning("Top50 Entity Linking Accuracy: {}".format(100 * top50 / total))
    my_logger.warning("Top100 Entity Linking Accuracy: {}".format(100 * top100 / total))

    f1.close()


convert('valid')
convert('test')

my_logger.warning("\n  Program End \n")