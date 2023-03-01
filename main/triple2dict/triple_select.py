from util import www2fb, clean_uri
import random

NUM=70000000
triple=set()

with open('freebase-FB2M.txt','r',encoding='utf-8') as f:
    for line in f:
        items = line.strip().split("\t")
        if len(items) != 3:
            print("ERROR: line - {}".format(line))
            continue

        subject = www2fb(items[0])
        predicate = www2fb(items[1])
        object = www2fb(items[2])
        triple.add('{}\t{}\t{}\n'.format(subject[3:],predicate[3:],object[3:]))
print(len(triple))

with open('/data/hn/fb_triple/fb_triple.txt','r',encoding='utf-8') as f:
    for line in f:
        j = random.randrange(10)
        if j<4:
            triple.add(line)
            if len(triple)==NUM:
                break
print(len(triple))

with open('triple_70M.txt','w',encoding='utf-8') as f:
    for i in triple:
        f.write(i)

        
        