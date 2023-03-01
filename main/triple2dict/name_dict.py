import pickle

names=dict()
num=0
en_large=pickle.load(open('/data/hn/BuboQA/indexes/entity.pkl','rb'))
for i in en_large.values():
    num+=1
    if num%10000==0:
        print(num)
    for j in i:
        if names.get(j[0]) is None:
            names[j[0]]=[j[1]]
        else:
            if j[1] not in names[j[0]]:
                names[j[0]].append(j[1])

en_set=set()
with open('triple_70M.txt','r',encoding='utf-8') as f:
    for i in f:
        items=i.strip().split('\t')
        en_set.add(items[0])
        
print(len(en_set))

f=open('en_name_error.txt','w',encoding='utf-8')
names_new=dict()
for i in en_set:
    if names.get(i) is not None:
        names_new[i]=names[i]
    else:
        f.write(i)
        f.write('\n')
f.close()

with open('names.pkl', 'wb') as f:
    pickle.dump(names_new, f)

        