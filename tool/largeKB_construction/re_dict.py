import pickle
from query_interface import query_en_relation
import logging

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',level=logging.DEBUG)
entity=[]
with open('surface_map_file_freebase_complete_all_mention','r') as f:
    for i, line in enumerate(f):
        if i % 1000000 == 0:
            logging.info(i)
        items = line.strip().split("\t")
        if len(items) != 3:
            print("ERROR: line - {}".format(line))
            continue

        entity_mid = items[2]
        entity_type = items[1]
        entity_name = items[0]
        entity.append(entity_mid)
        break

f1=open('error.txt','w',encoding='utf-8')
logging.info('yes')
num=0
redict=dict()

#redict=dict()
for i in entity:
    try:
        a=query_en_relation(i)
        a=set(a)            
        redict[i]=a
        num+=1
        if num%1000==0:
            logging.info(num)
    except:
        f1.write(i)
        f1.write('\n')
        continue


with open('redict.pkl', 'wb') as f:
    pickle.dump(redict, f)


print(redict)