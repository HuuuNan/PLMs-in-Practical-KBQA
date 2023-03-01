import pickle
import logging
from query_interface import query_en_triple

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',level=logging.DEBUG)
entity=pickle.load(open('entity3.pkl','rb'))
f=open('entity.txt','w',encoding='utf-8')
f1=open('fb_triple.txt','w',encoding='utf-8')
f2=open('error.txt','w',encoding='utf-8')
en_set=set()
num=0

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
    try:
        temp=query_en_triple(i)
        for j in temp:
            line = "{}\t{}\t{}\n".format(j[0],j[1],j[2])
            f1.write(line)
    except:
        f2.write(i)
        f2.write('\n')

f.close()
f1.close()
f2.close()


    
