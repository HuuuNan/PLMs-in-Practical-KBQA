import pandas as pd
import json
import time
import os
import sys
sys.path.append("..")
from mylogger import mylog
file_name = os.path.basename(__file__)
file_name = file_name[0:len(file_name)-3]

def convert(ques,label):
    if 'I' not in label or 'O' not in label:
        return ques
    result=''
    ques=ques.split(' ')
    label=label.split(' ')
    index=0
    while index!=len(ques):
        if label[index]=='O':
            result+=ques[index]
            index+=1
        else:
            result+='<e>'
            #找到第一个不为I的地方
            for i in range(index,len(ques)):
                if label[i]=='O':
                    index=i
                    break
            #考虑句子末尾的情况
            if label[index]=='I':
                return result
        result+=' '
    return result[:-1]

MODEL='roberta-base'
SCALE='medium2'


# set logger
my_logger = mylog.log_creater('./log', file_name+'_'+MODEL+'_'+SCALE+'-out')
my_logger.warning("\n new process start  \n")
my_logger.warning(MODEL+'-'+SCALE)


os.makedirs(SCALE+'/'+MODEL, exist_ok=True)
test=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
'''
name_list=[]
with open('../实体识别/'+MODEL+'_query/query.test','r',encoding='utf-8') as f:
    line=f.readline()

    while line!='':
        print(line)
        line=line.split(' %%%% ')
        print(line)
        name_list.append(line[1].strip())
        line=f.readline()
'''
tagall=[]
with open('../实体识别/test/'+MODEL+'_label.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        tagall.append(line)
        line=f.readline()

f=open('test_ques.txt','w',encoding='utf-8')
total=0
total_dict=dict()
for i in range(0,len(test)):
    temp=dict()
    tempt=time.time()
    ques=convert(test["question"][i],tagall[i])
    total=total+time.time()-tempt
    f.write(ques)
    f.write('\n')
    temp['question']=ques
    temp['entity']=test['entity_mid'][i]
    temp['origin_question']=test['question'][i]
    relation=test['relation'][i]
    relation=relation.replace('.',' ')
    relation=relation.replace('_',' ')
    relation=relation[3:len(relation)]
    temp['relation']=relation
    total_dict[test['lineid'][i]]=temp
f.close()

with open(SCALE+'/'+MODEL+'/test_dict.json', 'w') as f:
    json.dump(total_dict, f,ensure_ascii=False)

my_logger.warning('测试集替换头实体耗时：{}'.format(total))