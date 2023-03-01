

from __future__ import absolute_import, division, print_function
import argparse
# import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import sys
from datetime import datetime
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss, MarginRankingLoss
from argparse import ArgumentParser
from pytorch_pretrained_bert.optimization import BertAdam
from logger_config import get_logger
import torch.nn as nn
from transformers import AutoTokenizer
from predict_my import predict_one
import pandas as pd
import time

# personal package
from field import *
from bert_function import *
from model_hn import *
from args import get_args
from data import Data

import sys
sys.path.append("..")
from mylogger import mylog
file_name = os.path.basename(__file__)
file_name = file_name[0:len(file_name)-3]

# neg sample function
def train_neg_sample(negs, neg_size):
    new_negs = []
    for (subwords, mask) in negs:
        l = len(mask)
        try:
            index = random.sample([i for i in range(l)], neg_size)
        except:
            from pdb import set_trace
            set_trace()
        new_negs.append(([subwords[i] for i in index], [mask[i] for i in index]))
    return new_negs

# train data batchlize
def train_batchlize(train_dataset_question, train_dataset_gold, train_dataset_negs, batch_size, neg_size):
    batches_train_question = data_batchlize(batch_size, train_dataset_question)
    batches_train_gold = data_batchlize(batch_size, train_dataset_gold)

    all_neg0, all_neg1 = [], []
    # ƴ����
    for one_neg in train_dataset_negs:  # case_num,((neg_size,L),(neg_size, L))
        all_neg0.extend(one_neg[0])
        all_neg1.extend(one_neg[1])
    new_train_dataset_negs = (all_neg0, all_neg1)

    batches_train_negs = data_batchlize(batch_size * neg_size, new_train_dataset_negs)

    for i, batch_neg in enumerate(batches_train_negs):
        new_batch_neg = []
        for one_data in batch_neg:
            case_nums, max_seq_len = one_data.shape
            one_data = one_data.reshape(neg_size, -1, max_seq_len)
            new_batch_neg.append(one_data)
        batches_train_negs[i] = tuple(new_batch_neg)
    return (batches_train_question, batches_train_gold, batches_train_negs)


def data_batchlize(batch_size, data_tuple):
    '''
    give a tuple, return batches of data
    '''
    (subwords, mask) = data_tuple

    batches_subwords, batches_mask = [], []

    indexs = [i for i in range(len(subwords))]
    start = 0
    start_indexs = []
    while start <= len(indexs) - 1:
        start_indexs.append(start)
        start += batch_size

    start = 0
    for start in start_indexs:
        cur_indexs = indexs[start:start + batch_size]
        cur_subwords = [subwords[i] for i in cur_indexs]
        cur_mask = [mask[i] for i in cur_indexs]

        maxlen_i, maxlen_j = 0, 0
        for i, j in zip(cur_subwords, cur_mask):
            maxlen_i, maxlen_j = max(maxlen_i, len(i)), max(maxlen_j, len(j))
        batch_a, batch_b = [], []
        for a, b in zip(cur_subwords, cur_mask):
            batch_a.append([i for i in a] + [0] * (maxlen_i - len(a)))
            batch_b.append([i for i in b] + [0] * (maxlen_j - len(b)))
        batches_subwords.append(torch.LongTensor(batch_a))
        batches_mask.append(torch.LongTensor(np.array(batch_b)))
        
    return [item for item in zip(batches_subwords, batches_mask)]

def train(args, bert_field, model):
    Dataset = Data(args)
    # datasets
    train_rawdata = Dataset.load('train')
    valid_rawdata = Dataset.load('valid')

    (train_rawdata_questions, train_rawdata_gold, train_rawdata_neg) = train_rawdata
    (valid_rawdata_questions, valid_rawdata_gold, valid_rawdata_neg) = valid_rawdata
    train_dataset_question = Dataset.numericalize(bert_field, train_rawdata_questions)
    train_dataset_gold = Dataset.numericalize(bert_field, train_rawdata_gold)
    train_dataset_negs = []
    for one_neg in train_rawdata_neg:
        train_dataset_neg = Dataset.numericalize(bert_field,
                                                 one_neg)  # train_dataset_neg is a tuple(subwords, lens, mask)
        train_dataset_negs.append(train_dataset_neg)
    print('train data loaded!')

    if args.neg_fix:
        sample_train_dataset_negs = train_neg_sample(train_dataset_negs, args.neg_size)
        train_data = train_batchlize(train_dataset_question, train_dataset_gold, sample_train_dataset_negs,
                                     args.batch_size, args.neg_size)
        print("train data batchlized............")

    valid_dataset_question = Dataset.numericalize(bert_field, valid_rawdata_questions)
    valid_dataset_gold = Dataset.numericalize(bert_field, valid_rawdata_gold)
    valid_dataset_negs = []
    for index, one_neg in enumerate(valid_rawdata_neg):
        if not one_neg:
            print('no neg paths', index)
        valid_dataset_neg = Dataset.numericalize(bert_field, one_neg)
        valid_dataset_negs.append(valid_dataset_neg)

    valid_dataset = (valid_dataset_question, valid_dataset_gold, valid_dataset_negs)
    print('valid data loaded!')

    # num of train steps
    print('train examples', len(train_rawdata_questions))
    num_train_steps = int(
        len(train_rawdata_questions) / args.batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    # loss function
    criterion = MarginRankingLoss(margin=args.margin)

    # train params
    patience = args.patience
    num_train_epochs = args.num_train_epochs
    iters_left = patience
    best_precision = 0
    num_not_improved = 0
    global_step = 0

    # logger.warning('\nstart training:%s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("start training!")
    start_t=time.time()
    # train and evaluate
    for epoch in range(args.num_train_epochs):
     
        # batchlize
        if not args.neg_fix:
            sample_train_dataset_negs = train_neg_sample(train_dataset_negs, args.neg_size)
            train_data = train_batchlize(train_dataset_question, train_dataset_gold, sample_train_dataset_negs,
                                         args.batch_size, args.neg_size)
            print("train data batchlized............")

        train_right = 0
        train_total = 0
        # ��ӡ
        print('start time')
        start_time = datetime.now()
        
        my_logger.warning('\nstart training:%s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(start_time)

        model.train()
        optimizer.zero_grad()
        loss_epoch = 0  # ���ε�������loss
        (batches_train_question, batches_train_gold, batches_train_negs) = train_data
        for step, (batch_train_question, batch_train_gold, batch_train_negs) in enumerate(
                zip(tqdm(batches_train_question), batches_train_gold, batches_train_negs)):
            if args.no_cuda == False:
                batch_train_question = (t.cuda() for t in batch_train_question)
                batch_train_gold = (t.cuda() for t in batch_train_gold)
                batch_train_negs = (t.cuda() for t in batch_train_negs)

            else:
                batch_train_question = (t for t in batch_train_question)
                batch_train_gold = (t for t in batch_train_gold)
                batch_train_negs = (t for t in batch_train_negs)

            # loss = model(batch_train_question, batch_train_gold, batch_train_negs)

            scores = model(batch_train_question, batch_train_gold, batch_train_negs)

            (pos_score, neg_scores) = scores
            pos_score = pos_score.expand_as(neg_scores).reshape(-1)
            neg_scores = neg_scores.reshape(-1)
            assert len(pos_score) == len(neg_scores)
            ones = torch.ones(pos_score.shape)
            if args.no_cuda==False:
                ones = ones.cuda()

            loss = criterion(pos_score, neg_scores, ones)


            # evaluate train
            result = (torch.sum(pos_score.reshape(-1, args.neg_size) > neg_scores.reshape(-1, args.neg_size),
                                -1) == args.neg_size).cpu()

            train_right += torch.sum(result).item()
            train_total += len(result)
            # print(loss.item(), train_right/train_total)


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            loss_epoch += loss
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        # ��ӡ
        end_time = datetime.now()
        my_logger.warning('\ntrain epoch %d time span:%s' % (epoch, end_time - start_time))
        print('train loss', loss_epoch.item())
        my_logger.warning('train loss:%f' % loss_epoch.item())
        print('train result', train_right, train_total, 1.0 * train_right / train_total)
        my_logger.warning(('train result', train_right, train_total, 1.0 * train_right / train_total))

        # ����
        my_logger.warning("---------------------evaluate------------------")
        right, total, precision = evaluate(args, model, valid_dataset, valid_rawdata, epoch)
        # right, total, precision = 0, 0, 0.0

        # ��ӡ
        print('valid result', right, total, precision)
        print('epoch time')
        print(datetime.now())
        print('*' * 20)
        my_logger.warning("epoch:%d\t" % epoch + "dev_Accuracy-----------------------%d/%d=%f\n" % (right, total, precision))
        end_time = datetime.now()
        my_logger.warning('dev epoch %d time span:%s' % (epoch, end_time - start_time))

        if precision > best_precision:
            best_precision = precision
            iters_left = patience
            print("epoch %d saved\n" % epoch)
            my_logger.warning("epoch %d saved\n" % epoch)
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            save_path=SCALE+'/'+MODEL+'/pytorch_model.bin'
            if not os.path.exists(SCALE+'/'+MODEL):
                os.makedirs(SCALE+'/'+MODEL)
            output_model_file = save_path
            torch.save(model_to_save.state_dict(), output_model_file)
        else:
            iters_left -= 1
            if iters_left == 0:
                break
    end_t=time.time()
    my_logger.warning('finish training!')
    my_logger.warning('training time: {}'.format(end_t-start_t))
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    save_path=SCALE+'/'+MODEL+'/final_model.bin'
    if not os.path.exists(SCALE+'/'+MODEL):
        os.makedirs(SCALE+'/'+MODEL)
    output_model_file = save_path
    torch.save(model_to_save.state_dict(), output_model_file)
    
    return precision


def evaluate(args, model, valid_dataset, valid_rawdata, epoch):
    model.eval()
    (valid_dataset_question, valid_dataset_gold, valid_dataset_negs) = valid_dataset
    (q_valid1, q_valid2) = valid_dataset_question
    (gold_valid1, gold_valid2) = valid_dataset_gold
    right = 0
    total = 0
    for index, (q1, q2, gold1, gold2, negs) in enumerate(
            zip(tqdm(q_valid1), q_valid2, gold_valid1, gold_valid2, valid_dataset_negs)):
        q = (q1, q2)
        gold = (gold1, gold2)
        batches_negs = data_batchlize(args.eval_batch_size, negs)
        pos_score, all_scores = model.cal_score(q, batches_negs, pos=gold)

        total += 1
        if len(all_scores) == torch.sum(pos_score > all_scores):
            right += 1
    return right, total, 1.0 * right / total

if __name__ == "__main__":
    #训练前调整参数

    args = get_args()
    MODEL='gpt2'
    SCALE='small'

    global my_logger
    my_logger = mylog.log_creater('./log', file_name + '_' + MODEL + '_' + SCALE + '-out')
    my_logger.warning("\n new process start  \n")
    my_logger.warning(MODEL + '-' + SCALE)


    os.makedirs(SCALE+'/'+MODEL,exist_ok=True)
    
    args.model=MODEL

    # set the path to pathranking folder.
    # modified by mdh
    args.train_file='../pathranking/'+SCALE+'/train.json'
    args.valid_file='../pathranking/'+SCALE+'/valid.json'

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # tokenize
    tokenizer=AutoTokenizer.from_pretrained("../pretrain/"+MODEL)
    bert_field = BertCharField('BERT', tokenizer=tokenizer)
    print("loaded tokenizer")

    # model
    model = Bert_Comparing(args)
    
    if args.no_cuda == False:
        model.cuda()

    print(model)
    train(args, bert_field, model)