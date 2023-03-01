tempid=''
correct=0
with open('small/lstm_results/test.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.split(' %%%% ')
        if tempid!=line[0]:
            tempid=line[0]
            if line[2]=='1':
                correct+=1
        line=f.readline()
        
print(correct/21687)