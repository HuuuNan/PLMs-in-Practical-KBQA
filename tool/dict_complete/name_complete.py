import pickle
from query_interface import query_med

a=pickle.load(open('names_2M.pkl','rb'))
f1=open('error.txt','w',encoding='utf-8')
f2=open('medicine.txt','w',encoding='utf-8')

num=0
for i in a.items():
    if num%1000==0:
        print(num)
    num+=1
    try:
        temp=query_med(i[0][3:])
    except:
        f1.write(i[0])
        f1.write('\n')
        continue
    if len(temp)!=0:
        value=i[1]
        for j in temp:
            if j not in value:
                value.append(j)
            f2.write(i[0])
            f2.write('\t')
            f2.write(j)
            f2.write('\n')
        a[i[0]]=value

with open('names_2M1.pkl', 'wb') as f:
    pickle.dump(a, f)
    
f1.close()
f2.close()