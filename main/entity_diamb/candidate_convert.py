import json
import os

MODEL='bert-base-uncased'
SCALE='small'

os.makedirs('data/'+SCALE+'/'+MODEL, exist_ok=True)

def convert(candidate_txt,candidate_json):
    alljson=dict()
    with open(candidate_txt,'r',encoding='utf-8') as f:
        for line in f.readlines():
            candict=[]
            line=line.strip().split(' %%%% ')
            index=line[0].split('-')[-1]
            canid=set()
            for i in range(1,len(line)):
                temp=line[i].split('\t')
                tempdict=dict()
                if SCALE=='small':
                    tempdict['id']=temp[0][3:]
                else:
                    tempdict['id']=temp[0]
                if tempdict['id'] in canid:
                    continue
                canid.add(tempdict['id'])
                tempdict['name']=temp[1]
                candict.append(tempdict)
            alljson[index]=[candict]
    json.dump(alljson,open(candidate_json,'w',encoding='utf-8'),ensure_ascii=False)

convert('../entity_linking/'+SCALE+'/'+MODEL+'/train-h100.txt','data/'+SCALE+'/'+MODEL+'/train_candidate.json')
convert('../entity_linking/'+SCALE+'/'+MODEL+'/valid-h100.txt','data/'+SCALE+'/'+MODEL+'/valid_candidate.json')
convert('../entity_linking/'+SCALE+'/'+MODEL+'/test-h100.txt','data/'+SCALE+'/'+MODEL+'/test_candidate.json')

