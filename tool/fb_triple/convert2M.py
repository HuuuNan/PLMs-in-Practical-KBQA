from util import www2fb, clean_uri
import pickle

f1=open('2Mtriple.txt','w',encoding='utf-8')
with open('freebase-FB2M.txt', 'r') as f:
    for i, line in enumerate(f):
        if i % 1000000 == 0:
            print("line: {}".format(i))
  
        items = line.strip().split("\t")
        if len(items) != 3:
            print("ERROR: line - {}".format(line))
  
        subject = www2fb(items[0])
        predicate = www2fb(items[1])
        object = www2fb(items[2])
        f1.write('{}\t{}\t{}\n'.format(subject,predicate,object))

f1.close()