import pickle
entity=pickle.load(open('entity.pkl','rb'))
for i in entity.items():
    temp=set()
    for j in i[1]:
        temp.add((j[0],j[1]))
    entity[i[0]]=temp
    
with open('entity1.pkl', 'wb') as f:
    pickle.dump(entity, f)

