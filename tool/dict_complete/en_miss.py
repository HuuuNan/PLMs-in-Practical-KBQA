import pickle
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
en_new=set()
en_old=set()
a=pickle.load(open('entity.pkl','rb'))
for i in a.items():
    for j in i[1]:
        en_new.add(j[0])

num=0
with open('en_miss.txt','w',encoding='utf-8') as f:
    b=pickle.load(open('entity_2M.pkl','rb'))
    for i in b.items():
        num+=1
        if num%1000==0:
            logging.info(num)
        for j in i[1]:
            size1=len(en_new)
            temp=j[0][3:]
            en_new.add(temp)
            size2=len(en_new)
            if size1!=size2:
                f.write(temp)
                f.write('\t')
                f.write(j[1])
                f.write('\n')
                

        