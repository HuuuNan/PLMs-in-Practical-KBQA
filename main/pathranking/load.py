import pickle

with open('../indexes/reachability_2M.pkl', 'rb') as handler:
    index = pickle.load(handler)

relation=set()
for i in index.values():
    for j in i:
        j=j[3:]
        j=j.replace('.',' ')
        j=j.replace('_',' ')
        relation.add(j)

with open('relation_small.txt','w',encoding='utf-8') as f:
    for i in relation:
        f.write(i)
        f.write('\n')

print("---start redict.pkl--- ")

with open('../indexes/redict.pkl', 'rb') as handler:
    index = pickle.load(handler)

print("---end redict.pkl--- ")

relation=set()
for i in index.values():
    for j in i:
        j=j.replace('.',' ')
        j=j.replace('_',' ')
        relation.add(j)

with open('relation_large.txt','w',encoding='utf-8') as f:
    for i in relation:
        f.write(i)
        f.write('\n')