import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import GPT2Ner,LukeNer
from data_load import NerDataset, pad, VOCAB
import os
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
from seqeval.metrics import precision_score,recall_score,f1_score
import time

def train(model, iterator, optimizer, criterion,epoch,epoch_all):
    model.train()
    loop = tqdm(enumerate(iterator),total=len(iterator))
    #display loss
    loss_cur=0
    for step, batch in loop:
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()
        loss_cur+=loss.item()

        optimizer.step()
        
        loop.set_description('Epochs {}/{}. Running Loss: {}'.format(epoch,epoch_all,round(loss_cur/(step+1),4)))

def eval(args,model, iterator,output,stage):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    
    iterator = tqdm(iterator, desc="Running Evaluation")
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save    
    ques_token=[]
    ques_sen=[]
    gold=[]
    pred=[]
    for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
        y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
        preds = [idx2tag[hat] for hat in y_hat]
        assert len(preds)==len(words.split())==len(tags.split())
        ques_token.append(words.split()[1:-1])
        gold.append(tags.split()[1:-1])
        pred.append(preds[1:-1])
        ques_sen.append(' '.join(words.split()[1:-1]))   
    
    # output prediction result
    if output:
        data=pd.read_table('mydata/'+stage+'.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
        lineid=list(data['lineid'])
        question=list(data['question'])
        #match lineid
        index=0
        with open(args.result+'/query.'+stage,'w',encoding='utf-8') as f:
            for a,b,c in zip(ques_token,pred,ques_sen):
                while c!=question[index]:
                    index+=1
                tempid=lineid[index]
                f.write(tempid+' %%%%')
                for j in range(0,len(b)):
                    if b[j]=='I':
                        f.write(' ')
                        f.write(a[j])
                f.write('\n')  
    
    p=precision_score(gold, pred)
    r=recall_score(gold, pred)
    f=f1_score(gold, pred)
    return p,r,f

if __name__=="__main__":
    MODEL='gpt2'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--trainset", type=str, default="data/train.txt")
    parser.add_argument("--validset", type=str, default="data/valid.txt")
    parser.add_argument("--testset", type=str, default="data/test.txt")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--model_path", default='pretrain/'+MODEL)
    parser.add_argument('--weight_decay',type=float, default=0)
    args = parser.parse_args()
    
    args.checkpoint=MODEL+'_output'
    args.result=MODEL+'_query'    
    os.makedirs(args.checkpoint,exist_ok=True)
    os.makedirs(args.result,exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained('pretrain/'+MODEL, do_lower_case=False)
    # gpt2 has no pad_token, use eos_token instead
    if MODEL=='gpt2':
        tokenizer.pad_token = tokenizer.eos_token

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if MODEL=='gpt2':
        model = GPT2Ner(args.model_path, len(VOCAB), device).cuda()
    if MODEL=='luke-base':
        model = LukeNer(args.model_path, len(VOCAB), device).cuda()

    train_dataset = NerDataset(args.trainset,tokenizer)
    eval_dataset = NerDataset(args.validset,tokenizer)
    test_dataset = NerDataset(args.testset,tokenizer)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr = args.lr,weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_f1=0
    iter_not_improve=0
    print('*'*20,'Start Training','*'*20)
    
    start=time.time()
    for epoch in range(0, args.epochs):
        train(model, train_iter, optimizer, criterion,epoch,args.epochs)

        print('*'*20,'Start Evaluation at Epoch {}'.format(str(epoch)),'*'*20)
        precision, recall, f1 = eval(args,model, eval_iter,False,'valid')
        print('*'*20,'Epoch {} Eval Result'.format(str(epoch)),'*'*20)
        print("Precision:{:10.6f}%".format(100. * precision))
        print("Recall:{:10.6f}%".format(100. * recall))
        print("F1 Score:{:10.6f}%".format(100. * f1))
        if f1>best_f1:
            best_f1=f1
            iter_not_improve=0
            torch.save(model.state_dict(), args.checkpoint+'/pytorch_model.bin')
            print('*'*20,'Save Model at Epoch {}'.format(str(epoch)),'*'*20)
        else:
            iter_not_improve+=1
            if iter_not_improve > args.patience:
                print('*'*20,'Patience of {} Steps Reach. Stop Training at Epoch {}'.format(str(args.patience),str(epoch)),'*'*20)
                break
            else:
                print('*'*20,'No Improvement. Current Patience Step {}'.format(str(iter_not_improve)),'*'*20)
    
    end=time.time()
    print('*'*20,'Training Time:',end-start,'*'*20)
    
    # load best model
    model.load_state_dict(torch.load(args.checkpoint+'/pytorch_model.bin'))
    model=model.cuda()
    
    # predict on valid
    print('*'*20,'Start Prediction on Valid','*'*20)
    start=time.time()
    P,R,F = eval(args,model, eval_iter,True,'valid')
    end=time.time()
    print('*'*20,'Valid Prediction Result','*'*20)
    print("Precision:{:10.6f}%".format(100. * P))
    print("Recall:{:10.6f}%".format(100. * R))
    print("F1 Score:{:10.6f}%".format(100. * F))
    print("Prediction on Valid Time:",end-start)
    
    # prediction on test
    print('*'*20,'Start Prediction on Test','*'*20)
    start=time.time()
    P,R,F = eval(args,model, test_iter,True,'test')
    end=time.time()
    print('*'*20,'Test Prediction Result','*'*20)
    print("Precision:{:10.6f}%".format(100. * P))
    print("Recall:{:10.6f}%".format(100. * R))
    print("F1 Score:{:10.6f}%".format(100. * F))
    print("Prediction on Test Time:",end-start)

