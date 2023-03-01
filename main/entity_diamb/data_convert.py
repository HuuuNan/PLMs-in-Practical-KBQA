import pandas as pd
import json

train_csv=pd.read_table('data/train.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
valid_csv=pd.read_table('data/valid.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
test_csv=pd.read_table('data/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])

def convert(csvfile,jsonfile):
    datalist=[]
    idlist=list(csvfile['lineid'])
    queslist=list(csvfile['question'])
    midlist=list(csvfile['entity_mid'])
    for i in range(0,len(idlist)):
        temp=dict()
        temp['qid']=i+1
        temp['question']=queslist[i]
        temp['gold']=midlist[i]
        datalist.append(temp)
    json.dump(datalist,open(jsonfile,'w',encoding='utf-8'),ensure_ascii=False)
    
convert(train_csv,'data/train_data.json')
convert(valid_csv,'data/valid_data.json')
convert(test_csv,'data/test_data.json')