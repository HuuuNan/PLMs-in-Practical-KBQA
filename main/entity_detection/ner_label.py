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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
TYPE="albert"
MODEL="albert-base-v2"

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
model_args.output_dir='test/'+MODEL+'_output/'
model_args.overwrite_output_dir=True
model_args.evaluate_during_training=True
model_args.use_early_stopping=True
model_args.early_stopping_consider_epochs=True
model_args.early_stopping_patience=10
model_args.best_model_dir='test/'+MODEL+'_output/best_model/'

custom_labels = ["O", "I"]

model = NERModel(TYPE,MODEL+'_output', args=model_args, labels=custom_labels,use_cuda=True)
result, model_outputs, preds_list = model.eval_model(test_data)
test_csv=pd.read_table('../mydata/test.txt', header=None, names=["lineid", "entity_mid", "entity_name", "relation", "object", "question", "tags"])
truth=test_csv['tags']

P, R, F = evaluation_my(truth, preds_list, type=False)
print("{} Precision: {:10.6f}% Recall: {:10.6f}% F1 Score: {:10.6f}%".format("Dev", 100. * P, 100. * R,100. * F))

test=[]
with open('./test_sentence.txt','r',encoding='utf-8') as f:
    line=f.readline()
    while line!='':
        line=line.strip()
        test.append(line)
        line=f.readline()

with open('test/'+MODEL+'_label.txt','w',encoding='utf-8') as f:
    for i in range(0,len(preds_list)):
        tag_list=preds_list[i]
        tag_str=tag_list[0]
        for j in range(1,len(tag_list)):
            tag_str=tag_str+' '+tag_list[j]
        f.write(tag_str)
        f.write('\n')