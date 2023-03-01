import pickle

f=open('entity1.txt','w',encoding='utf-8')
entity=pickle.load(open('entity3.pkl','rb'))
en_set=set()
for i in entity.values():
    for j in i:
        en_set.add(j[0])
        num+=1
        if num%1000000==0:
            logging.info(num)
            
for num1,i in enumerate(en_set):
    if num1%1000000==0:
        logging.info(num1)
    f.write(i)
    f.write('\n')