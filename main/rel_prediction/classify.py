import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import time
import numpy as np
from argparse import ArgumentParser
from transformers import AutoTokenizer
import pickle
from torch.utils.data import DataLoader
import pandas as pd
import random
from tqdm import tqdm
from model import Lukeforclass,GPT2forclass
import torch.nn.functional as F
import sys
sys.path.append("..")
from mylogger import mylog
file_name = os.path.basename(__file__)
file_name = file_name[0:len(file_name)-3]

class GetLoader(torch.utils.data.Dataset):
	  # init function
    def __init__(self, data_id,data_mask,data_label):
        self.data_id = data_id
        self.data_mask = data_mask
        self.label = data_label
    # return id, mask, label
    def __getitem__(self, index):
        data_id = self.data_id[index]
        data_mask = self.data_mask[index]
        label = self.label[index]
        return data_id,data_mask,label
    # return length
    def __len__(self):
        return len(self.label)

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--batch_size_eval', type=int, default=16)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument('--maxlength', type=int, default=128)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
parser.add_argument('--seed', type=int, default=3435)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay',type=float, default=0)
args = parser.parse_args()

MODEL='gpt2'
SCALE='large'

# set logger
my_logger = mylog.log_creater('./log', file_name+'_'+MODEL+'_'+SCALE+'-out')
my_logger.warning("\n new process start  \n")
my_logger.warning(MODEL+'-'+SCALE)


args.model=MODEL
tokenizer=AutoTokenizer.from_pretrained('../pretrain/'+MODEL)
os.makedirs(SCALE+'/'+MODEL+'_output', exist_ok=True)
os.makedirs(SCALE+'/'+MODEL+'_results', exist_ok=True)
os.makedirs('tokenize/'+SCALE+'/'+MODEL, exist_ok=True)

# prepare all relations
exist=[]
with open('relation_'+SCALE+'.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        exist.append(line.strip())
        line=f.readline()

# set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

train = pd.read_table('../mydata/train.txt', header=None,
                      names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
valid = pd.read_table('../mydata/valid.txt', header=None,
                      names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
test = pd.read_table('../mydata/test.txt', header=None,
                     names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])

# prepare dataset
trainques=list(train['question'])
trainre=list(train['relation'])
trainlabel=[]
for i in trainre:
    # deal with relation start with fb: in small scale
    if SCALE!='small':
        temp=i[3:]
    else:
        temp=i
    if temp in exist:
        trainlabel.append(exist.index(temp))
    else:
        trainlabel.append(len(exist))

validques=list(valid['question'])
validre=list(valid['relation'])
validlabel=[]
for i in validre:
    # deal with relation start with fb: in small scale
    if SCALE!='small':
        temp=i[3:]
    else:
        temp=i
    if temp in exist:
        validlabel.append(exist.index(temp))
    else:
        validlabel.append(len(exist))

testques=list(test['question'])
testre=list(test['relation'])
testlabel=[]
for i in testre:
    # deal with relation start with fb: in small scale
    if SCALE!='small':
        temp=i[3:]
    else:
        temp=i
    if temp in exist:
        testlabel.append(exist.index(temp))
    else:
        testlabel.append(len(exist))

args.label=len(exist)
if MODEL=='gpt2':
    model = GPT2forclass(args)
if MODEL=='luke-base':
    model = Lukeforclass(args)
print(model)
model=model.cuda()
parameter = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

# gpt2 has no pad_token, use eos_token instead
if MODEL=='gpt2':
    tokenizer.pad_token = tokenizer.eos_token

# tokenize
if not os.path.exists('tokenize/'+SCALE+'/'+MODEL+'/tokenize.pkl'):
    print('*'*20,'Start Tokenization','*'*20)
    trainques=tokenizer(trainques,padding=True, truncation=True,return_tensors="pt",max_length=args.maxlength)
    validques=tokenizer(validques,padding=True, truncation=True,return_tensors="pt",max_length=args.maxlength)
    testques=tokenizer(testques,padding=True, truncation=True,return_tensors="pt",max_length=args.maxlength)
    pickle.dump((trainques,validques,testques),open('tokenize/'+SCALE+'/'+MODEL+'/tokenize.pkl','wb'))
else:
    print('*'*20,'Tokenization has been Done. Load Directly','*'*20)
    trainques,validques,testques=pickle.load(open('tokenize/'+SCALE+'/'+MODEL+'/tokenize.pkl','rb'))

Mytrain=GetLoader(trainques['input_ids'].cuda(),trainques['attention_mask'].cuda(),torch.tensor(trainlabel).cuda())
Myvalid=GetLoader(validques['input_ids'].cuda(),validques['attention_mask'].cuda(),torch.tensor(validlabel).cuda())
Mytest=GetLoader(testques['input_ids'].cuda(),testques['attention_mask'].cuda(),torch.tensor(testlabel).cuda())

traindata = DataLoader(Mytrain, batch_size=args.batch_size, drop_last=False)
validdata = DataLoader(Myvalid, batch_size=args.batch_size_eval, drop_last=False)
testdata = DataLoader(Mytest, batch_size=args.batch_size_eval, drop_last=False)

# for early stopping
best_acc, iters_not_improved = 0, 0
patience = args.patience  
epoch = 0

print('*'*20,'Start Training','*'*20)
start = time.time()
for epoch in range(0, args.epochs):
    model.train()
    loop = tqdm(enumerate(traindata),total=len(traindata))
    #display loss
    loss_cur=0
    for step, batch in loop:
        optimizer.zero_grad()
        loss,score=model(batch[0],batch[1],batch[2])
        loss.backward()
        loss_cur+=loss.item()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()
        #display train information
        loop.set_description('Epochs {}/{}. Running Loss: {}'.format(epoch,args.epochs,round(loss_cur/(step+1),4)))
    
    # Eval during training
    acc=0
    correct=0
    num=0
    pred=[]
    model.eval()
    validdata = tqdm(validdata, desc="Running Evaluation")
    with torch.no_grad():
        for step, dev_batch in enumerate(validdata):
            loss,answer = model(dev_batch[0],dev_batch[1])
            label = dev_batch[2]
            correct+=torch.sum((torch.max(answer, 1)[1].view(label.size()).data == label.data)).item()
            num+=len(label)
    acc=correct/num
    print('*'*20,'Epoch {} Eval Accuracy: {}'.format(str(epoch),acc),'*'*20)
    
    # update model
    if acc > best_acc:
        best_acc = acc
        iters_not_improved = 0
        torch.save(model.state_dict(), SCALE+'/'+MODEL+'_output/pytorch_model.bin')
        print('*'*20,'Save Model at Epoch {}'.format(str(epoch)),'*'*20)
    else:
        iters_not_improved += 1
        if iters_not_improved > patience:
            print('*'*20,'Patience of {} Steps Reach. Stop Training at Epoch {}'.format(str(args.patience),str(epoch)),'*'*20)
            break
        else:
            print('*'*20,'No Improvement. Current Patience Step {}'.format(str(iters_not_improved)),'*'*20)

end=time.time()
print('*'*20,'Training Time:',end-start,'*'*20)
my_logger.warning("Training Time:{}".format(end-start))

# load best model
model.load_state_dict(torch.load(SCALE+'/'+MODEL+'_output/pytorch_model.bin'))
model=model.cuda()

# prediction on valid
model.eval()
print('*'*20,'Start Prediction on Valid','*'*20)
validdata = tqdm(validdata, desc="Running Evaluation")
model_outputs=[]
pred=[]
score=[]
start=time.time()
with torch.no_grad():
    for step, dev_batch in enumerate(validdata):
        loss,answer = model(dev_batch[0],dev_batch[1])
        model_outputs.extend(answer.cpu().detach().tolist())
end=time.time()
model_outputs = F.log_softmax(torch.tensor(model_outputs), dim=1).numpy()
indexlist=list(valid['lineid'])
correct = 0
retrieve = 0
with open(SCALE + '/' + MODEL + '_results' + '/valid.txt', 'w', encoding='utf-8') as f:
    for i in range(0, len(model_outputs)):
        truth = validlabel[i]
        # output: score, order: index according to score from large to small
        output = np.flipud(np.sort(model_outputs[i]))
        order = np.flipud(np.argsort(model_outputs[i]))
        if truth == order[0]:
            correct += 1
        for j in range(0, 5):
            # temp1: relation, temp2: score, temp3: correct or not
            temp1 = exist[order[j]]
            temp2 = output[j]
            if order[j] == truth:
                temp3 = 1
                retrieve += 1
            else:
                temp3 = 0
            f.write("{} %%%% {} %%%% {} %%%% {}\n".format(indexlist[i], temp1, temp3, temp2))
P = correct / len(validlabel)
R = retrieve / len(validlabel)
print('*'*20,'Valid Prediction Result','*'*20)
print("Precision:{:10.6f}%".format(100. * P))
print("Top5:{:10.6f}%".format(100. * R))
print("Prediction on Valid Time:",end-start)

# prediction on test
model.eval()
print('*'*20,'Start Prediction on Test','*'*20)
testdata = tqdm(testdata, desc="Running Evaluation")
model_outputs=[]
pred=[]
score=[]
start=time.time()
with torch.no_grad():
    for step, dev_batch in enumerate(testdata):
        loss,answer = model(dev_batch[0],dev_batch[1])
        model_outputs.extend(answer.cpu().detach().tolist())
end=time.time()
model_outputs = F.log_softmax(torch.tensor(model_outputs), dim=1).numpy()
indexlist=list(test['lineid'])
correct = 0
retrieve = 0
with open(SCALE + '/' + MODEL + '_results' + '/test.txt', 'w', encoding='utf-8') as f:
    for i in range(0, len(model_outputs)):
        truth = testlabel[i]
        # output: score, order: index according to score from large to small
        output = np.flipud(np.sort(model_outputs[i]))
        order = np.flipud(np.argsort(model_outputs[i]))
        if truth == order[0]:
            correct += 1
        for j in range(0, 5):
            # temp1: relation, temp2: score, temp3: correct or not
            temp1 = exist[order[j]]
            temp2 = output[j]
            if order[j] == truth:
                temp3 = 1
                retrieve += 1
            else:
                temp3 = 0
            f.write("{} %%%% {} %%%% {} %%%% {}\n".format(indexlist[i], temp1, temp3, temp2))
P = correct / len(testlabel)
R = retrieve / len(testlabel)
print('*'*20,'Test Prediction Result','*'*20)

my_logger.warning("\n  Result: \n")
my_logger.warning("Precision:{:10.6f}%".format(100. * P))
my_logger.warning("Retrieval Rate:{:10.6f}%".format(100. * R))
my_logger.warning("Prediction on Test Time:{}".format(end-start))

my_logger.warning("\n  Program End \n")