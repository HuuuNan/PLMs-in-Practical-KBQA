triple=set()
with open('fb_triple.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        triple.add(line)
        line=f.readline()

miss=set()
f1=open('triple_miss.txt','w',encoding='utf-8')
with open('2Mtriple.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        num1=len(triple)
        triple.add(line)
        num2=len(triple)
        if num1!=num2:
            f1.write(line)
            miss.add(line)
        line=f.readline()

f1.close()
with open('fb_triple.txt','a',encoding='utf-8') as f:
    for i in miss:
        f.write(i)            
         