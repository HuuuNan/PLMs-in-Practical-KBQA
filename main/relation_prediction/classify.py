import pandas as pd
import random
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import numpy as np
import os
import time
import psutil
from torchtext import data
import torch.nn.functional as F
import torch
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
TYPE='albert'
MODEL="albert-base-v2"
SCALE='large'

train=pd.read_table('../mydata/train.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])   
valid=pd.read_table('../mydata/valid.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])   
test=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])

if SCALE=='small':
    redict=pickle.load(open('../indexes/reachability_2M.pkl','rb'))
else:
    redict=pickle.load(open('../indexes/redict.pkl','rb'))
pre_start=time.time()
exist=set()
for i in redict.items():
    for j in i[1]:
        exist.add(j)
exist=list(exist)

tag=[]
for i in train['relation']:
    if i[3:] in exist:
        num=exist.index(i[3:])
        tag.append(num)
    else:
        tag.append(len(exist))
        
train_df=pd.DataFrame()
train_df['text']=train['question']
train_df['labels']=tag
print(train_df)
tag=[]
for i in valid['relation']:
    if i[3:] in exist:
        num=exist.index(i[3:])
        tag.append(num)
    else:
        tag.append(len(exist))

valid_df=pd.DataFrame()
valid_df['text']=valid['question']
valid_df['labels']=tag
        
tag=[]
for i in test['relation']:
    if i[3:] in exist:
        num=exist.index(i[3:])
        tag.append(num)
    else:
        tag.append(len(exist))
    
test_df=pd.DataFrame()
test_df['text']=test['question']
test_df['labels']=tag

# Configure the model
model_args = ClassificationArgs()
model_args.train_batch_size = 64
model_args.eval_batch_size = 16
model_args.n_gpu=1
model_args.save_best_model = True
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.save_optimizer_and_scheduler = False
model_args.save_steps = -1
model_args.evaluate_during_training_verbose=True
model_args.overwrite_output_dir=True
model_args.num_train_epochs=1000
model_args.output_dir=SCALE+'/'+MODEL+'_output/'
model_args.overwrite_output_dir=True
model_args.evaluate_during_training=True
model_args.use_early_stopping=True
model_args.early_stopping_consider_epochs=True
model_args.early_stopping_patience=10
model_args.best_model_dir=SCALE+'/'+MODEL+'_output/best_model/'
model_args.use_multiprocessing = False
model_args.dataloader_num_workers = 0
model_args.process_count = 1
model_args.use_multiprocessing_for_evaluation = False

# Create a MultiLabelClassificationModel
model = ClassificationModel(
    TYPE, '../pretrain/'+MODEL, num_labels=len(exist),use_cuda=True,args=model_args
)

if not os.path.exists(SCALE+'/'+MODEL+'_results'):
    os.mkdir(SCALE+'/'+MODEL+'_results')
    
# Train the model
time1=time.time()
model.train_model(train_df,eval_df=valid_df)
time2=time.time()
train_time=time2-time1

# Evaluate the model
print('evaluate on valid:')
time3=time.time()
result, model_outputs, wrong_predictions = model.eval_model(valid_df,verbose=True)
time4=time.time()
valid_time=time4-time3
model_outputs=torch.from_numpy(model_outputs)
model_outputs = F.log_softmax(model_outputs, dim=1)
model_outputs=model_outputs.numpy()
correct=0
retrieve=0
f=open(SCALE+'/'+MODEL+'_results'+'/valid.txt','w',encoding='utf-8')
for i in range(0,len(model_outputs)):
    truth=valid_df['labels'][i]
    index='valid-'+str(i+1)
    output=np.flipud(np.sort(model_outputs[i]))
    order=np.flipud(np.argsort(model_outputs[i]))
    if truth==order[0]:
        correct+=1
    for j in range(0,5):
        temp1=exist[order[j]]
        temp2=output[j]
        if order[j]==truth:
            temp3=1
            retrieve+=1
        else:
            temp3=0
        f.write("{} %%%% {} %%%% {} %%%% {}\n".format(index, temp1, temp3, temp2))  
f.close()
P = 1. * correct / len(valid_df)
print("Precision: ",P)
print("retrieved: {} out of {}".format(retrieve, len(valid_df)))
retrieval_rate = 1. * retrieve / len(valid_df)
print("Retrieval Rate: ",retrieval_rate)

#predict
print('evaluate on test:')
time5=time.time()
result, model_outputs, wrong_predictions = model.eval_model(test_df,verbose=True)
time6=time.time()
test_time=time6-time5

model_outputs=torch.from_numpy(model_outputs)
model_outputs = F.log_softmax(model_outputs, dim=1)
model_outputs=model_outputs.numpy()
correct=0
retrieve=0
f=open(SCALE+'/'+MODEL+'_results'+'/test.txt','w',encoding='utf-8')
for i in range(0,len(model_outputs)):
    truth=test_df['labels'][i]
    index='test-'+str(i+1)
    output=np.flipud(np.sort(model_outputs[i]))
    order=np.flipud(np.argsort(model_outputs[i]))
    if truth==order[0]:
        correct+=1
    for j in range(0,5):
        temp1=exist[order[j]]
        temp2=output[j]
        if order[j]==truth:
            temp3=1
            retrieve+=1
        else:
            temp3=0
        f.write("{} %%%% {} %%%% {} %%%% {}\n".format(index, temp1, temp3, temp2))  
f.close()

P = 1. * correct / len(test_df)
print("Precision: ",P)
print("retrieved: {} out of {}".format(retrieve, len(test_df)))
retrieval_rate = 1. * retrieve / len(test_df)
print("Retrieval Rate: ",retrieval_rate)
