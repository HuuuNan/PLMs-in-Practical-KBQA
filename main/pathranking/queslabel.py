#!/usr/bin/python3
# coding=gbk
def convert(ques,label):
    if 'I' not in label or 'O' not in label:
        return ques
    result=''
    ques=ques.split(' ')
    label=label.split(' ')
    index=0
    while index!=len(ques):
        if label[index]=='O':
            result+=ques[index]
            index+=1
        else:
            result+='<e>'
            #�ҵ���һ����ΪI�ĵط�
            for i in range(index,len(ques)):
                if label[i]=='O':
                    index=i
                    break
            #���Ǿ���ĩβ�����
            if label[index]=='I':
                return result
        result+=' '
    return result[:-1]

a=convert('which genre of album is harder ... ..faster ?','I O I I O I I O I')
print(a)