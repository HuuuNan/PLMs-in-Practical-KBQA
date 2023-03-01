import pandas as pd
import time
import sys
sys.path.append("..")
from mylogger import mylog

# set logger
my_logger = mylog.log_creater('./log', 'data_process_ques-out_mdh')

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

train=pd.read_table('../mydata/train.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
valid=pd.read_table('../mydata/valid.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
test=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])

total=0

ques_list=train['question']
name_list=train['entity_name']
tag_list=train["tags"]
with open('./train_ques.txt','w',encoding='utf-8') as f1:
    for i in range(0,len(ques_list)):
        ques=ques_list[i]
        name=name_list[i]
        tag=tag_list[i]
        temp=time.time()
        ques=convert(ques,tag)
        total=total+time.time()-temp   
        f1.write(ques)
        f1.write('\n')
     
ques_list=valid['question']
name_list=valid['entity_name']   
tag_list=valid["tags"]
with open('./valid_ques.txt','w',encoding='utf-8') as f1:
    for i in range(0,len(ques_list)):
        ques=ques_list[i]
        name=name_list[i]
        tag=tag_list[i]
        temp=time.time()
        ques=convert(ques,tag)
        total=total+time.time()-temp   
        f1.write(ques)
        f1.write('\n')
        
print('训练集替换头实体耗时：',total)
my_logger.info('训练集替换头实体耗时：{}'.format(total))

'''
ques_list=test['question']
name_list=test['entity_name']   
with open('./test_ques.txt','w',encoding='utf-8') as f1:
    for i in range(0,len(ques_list)):
        ques=ques_list[i].split()
        name=name_list[i].split()
        ques1=''
        if name[0] in ques:
            start=ques.index(name[0])
            end=ques.index(name[-1])
            ques[start]='#'+ques[start]
            ques[end]+='#'
            for j in ques:
                ques1+=j
                ques1+=' '
            ques1=ques1[0:len(ques1)-1]
        else:
            for j in ques:
                ques1+=j
                ques1+=' '
            ques1=ques1[0:len(ques1)-1]
        f1.write(ques1)
        f1.write('\n')
'''


