# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 23:30:28 2021

@author: Lenovo
"""

import logging
import os
import pandas as pd
from simpletransformers.ner import NERModel, NERArgs
from evaluation import evaluation_my
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TYPE="bert"
MODEL="bert-base-uncased"

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data = './train.txt'
valid_data = './valid.txt'
test_data='./test.txt'

# Configure the model
model_args = NERArgs()
model_args.train_batch_size = 64
model_args.eval_batch_size = 16
model_args.n_gpu=1
model_args.save_best_model = True
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.save_optimizer_and_scheduler = False
model_args.save_steps = -1
model_args.evaluate_during_training_verbose=False
model_args.overwrite_output_dir=True
model_args.num_train_epochs=1000
model_args.output_dir='./'+MODEL+'_output/'
model_args.overwrite_output_dir=True
model_args.evaluate_during_training=True
model_args.use_early_stopping=True
model_args.early_stopping_consider_epochs=True
model_args.early_stopping_patience=10
model_args.best_model_dir='./'+MODEL+'_output/best_model/'
model_args.evaluate_during_training_steps=-1

if not os.path.exists('./'+MODEL+'_query'):
    os.mkdir('./'+MODEL+'_query')

custom_labels = ["O", "I"]

model = NERModel(TYPE,'../pretrain/'+MODEL, args=model_args, labels=custom_labels,use_cuda=True)

# Train the model
time1=time.time()
model.train_model(train_data,eval_df=valid_data)
time2=time.time()
train_time=time2-time1

# Evaluate the model
print('evaluate on valid.txt')
time3=time.time()
result, model_outputs, preds_list = model.eval_model(valid_data)
time4=time.time()
valid_time=time4-time3

valid_csv=pd.read_table('../mydata/valid.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
truth=valid_csv['tags']

valid=[]
with open('./valid_sentence.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        valid.append(line)
        line=f.readline()

with open(MODEL+'_query/'+'query.valid','w',encoding='utf-8') as f:
    for i in range(0,len(preds_list)):
        f.write('valid-'+str(i+1))
        f.write(' %%%% ')
        tag=preds_list[i]
        sentence=valid[i].split(' ')
        for j in range(0,len(tag)):
            if tag[j]=='I':
                f.write(sentence[j])
                f.write(' ')
        f.write('\n')

print('evaluate on test.txt')
time5=time.time()
result, model_outputs, preds_list = model.eval_model(test_data)
time6=time.time()
test_time=time6-time5

test_csv=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
truth=test_csv['tags']

test=[]
with open('./test_sentence.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        test.append(line)
        line=f.readline()

with open(MODEL+'_query/'+'query.test','w',encoding='utf-8') as f:
    for i in range(0,len(preds_list)):
        f.write('test-'+str(i+1))
        f.write(' %%%% ')
        tag=preds_list[i]
        sentence=test[i].split(' ')
        for j in range(0,len(tag)):
            if tag[j]=='I':
                f.write(sentence[j])
                f.write(' ')
        f.write('\n')
        