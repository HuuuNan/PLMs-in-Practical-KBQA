import os

import argparse
import glob
import logging

import random
import sys
import timeit
from functools import partial
from os.path import join
from collections import OrderedDict
import pickle
import json
import time

import numpy as np
from numpy.lib.shape_base import expand_dims
import torch
from torch.utils import data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from components.model import CandidateRanking

import sys

sys.path.append("..")
from mylogger import mylog

file_name = os.path.basename(__file__)
file_name = file_name[0:len(file_name) - 3]

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from components.config import set_seed, to_list, register_args, validate_args, load_untrained_model
from components.dataset_utils import ListDataset
from components.disamb_dataset import (
    read_disamb_instances_from_entity_candidates,
    extract_disamb_features_from_examples,
    disamb_collate_fn,
    coverage_evaluation
)

from components.utils import mkdir_p, dump_json


# logger = logging.getLogger(__name__)

def train(args, train_dataset, model, tokenizer, redict):
    SCALE = args.train_file.split('/')[1]
    MODEL = args.train_file.split('/')[2]
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
        mkdir_p(args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_collate_fn = partial(disamb_collate_fn, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=train_collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max(args.warmup_steps, t_total * args.warmup_ratio)),
        num_training_steps=t_total
    )

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Warmup steps = %d", int(max(args.warmup_steps, t_total * args.warmup_ratio)))
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    # early stopping
    best_acc = 0
    patience = 0
    epoch = 0
    print('*' * 20, 'Start Training', '*' * 20)

    time1 = time.time()
    save_cost, train_time = 0.0, 0.0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[1],
                "attention_mask": batch[2],
                "sample_mask": batch[3],
                "labels": batch[4],
            }

            # del inputs["token_type_ids"]
            if args.model_type in ["roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]

            outputs = model(**inputs)

            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            # print("loss:", loss.data)


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log infomation
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    logs['epoch'] = _ + (step + 1) / len(epoch_iterator)
                    logs['learning_rate'] = scheduler.get_last_lr()[0]
                    logs['loss'] = (tr_loss - logging_loss) / args.logging_steps
                    logs['step'] = global_step
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("Training logs: {}".format(logs))
                    logging_loss = tr_loss
            
            if (step + 1) % args.eval_steps == 0:
                results, valid_time = evaluate(args, model, tokenizer, redict)
                for key, value in results.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                logger.info("Eval results: {}".format(dict(results)))
                if results['acc'] > best_acc:
                    save_start = time.time()
                    patience = 0
                    best_acc = results['acc']
                    output_dir = args.checkpoint_dir + '/' + SCALE + '/' + MODEL
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    torch.save(model_to_save.state_dict(), output_dir + '/pytorch_model.bin')
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    tmp_time = time.time() - save_start
                    save_cost += tmp_time
                else:
                    patience += 1
                    if patience > args.patience:
                        logger.info('Patience of {} Steps Reach. Stop Training'.format(str(args.patience)))
                        break
                    else:
                        logger.info('No Improvement. Current Patience Step {}'.format(patience))
        '''
        if args.evaluate_during_training:
            results,valid_time = evaluate(args, model, tokenizer, redict)
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            logger.info("Eval results: {}".format(dict(results)))
            if results['acc']>best_acc:
                save_start = time.time()
                patience=0
                best_acc=results['acc']
                output_dir = args.checkpoint_dir+'/'+SCALE+'/'+MODEL
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(model_to_save.state_dict(), output_dir+'/pytorch_model.bin')
                tokenizer.save_pretrained(output_dir)
  
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)
  
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)
                tmp_time = time.time() - save_start
                save_cost += tmp_time
            else:
                patience+=1
                if patience>args.patience:
                    # time2=time.time()
                    # train_time=time2-time1
                    logger.info('Patience of {} Steps Reach. Stop Training'.format(str(args.patience)))
                    break
                else:
                    logger.info('No Improvement. Current Patience Step {}'.format(patience))
        '''
    train_time = time.time() - time1
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    train_time = train_time - save_cost

    return global_step, tr_loss / global_step, train_time


def evaluate(args, model, tokenizer, redict, output_prediction=False):
    SCALE = args.train_file.split('/')[1]
    MODEL = args.train_file.split('/')[2]
    if 'valid' in args.predict_file:
        stage = 'valid'
    if 'test' in args.predict_file:
        stage = 'test'
    dataset, examples, token_time = load_and_cache_examples(args, tokenizer, redict, stage, output_examples=True, is_train = False)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=partial(disamb_collate_fn, tokenizer=tokenizer))

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = timeit.default_timer()

    all_pred_indexes = []
    all_labels = []
    all_scores = []
    time3 = time.time()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[1],
                "attention_mask": batch[2],
                "sample_mask": batch[3],
                "labels": batch[4],
            }
            # del inputs["token_type_ids"]
            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]
            output = model(**inputs)
            loss = output[0]
            # print(loss.data)
            logits = output[1]
            all_scores.extend(logits.cpu())
            pred_indexes = torch.argmax(logits, 1).detach().cpu()
        all_pred_indexes.append(pred_indexes)
        all_labels.append(batch[4].cpu())
    time4 = time.time()
    # add tokenize time
    valid_time = time4 - time3 + token_time
    all_pred_indexes = torch.cat(all_pred_indexes).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = np.sum(all_pred_indexes == all_labels) / len(all_pred_indexes)
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    coverage = coverage_evaluation(examples, dataset, all_pred_indexes)
    results = {'num problem': len(all_pred_indexes), 'acc': acc, 'cov': coverage}
    print('num problem ', len(all_pred_indexes), ' acc ', acc, ' cov ', coverage)
    result_dict = dict()
    # print(all_scores[0])
    # print(all_scores[1])

    if output_prediction:
        candict = json.load(
            open('data/' + SCALE + '/' + MODEL + '/' + stage + '_candidate.json', 'r', encoding='utf-8'))
        for feat, score in zip(dataset, all_scores):
            tempdict = dict()
            score = score.numpy()
            tempmin = np.min(score)
            tempmax = np.max(score)
            if tempmin == tempmax:
                score = np.ones(len(score))
            else:
                score = (score - np.min(score)) / (np.max(score) - np.min(score))
            for mid, cand_score in zip(candict[feat.pid][0], score):
                tempdict[mid["id"]] = float(cand_score)
            result_dict[feat.pid] = tempdict
        json.dump(result_dict,
                  open(args.output_dir + '/' + SCALE + '/' + MODEL + '/' + stage + '.json', 'w', encoding='utf-8'),
                  ensure_ascii=False)
    return results, valid_time


def load_and_cache_examples(args, tokenizer, redict, stage, output_examples=False, is_train=True):
    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    SCALE = args.train_file.split('/')[1]
    MODEL = args.train_file.split('/')[2]
    os.makedirs('feature_cache/' + SCALE + '/' + MODEL, exist_ok=True)
    cached_features_file = 'feature_cache/' + SCALE + '/' + MODEL + '/' + stage + '.bin'

    trian_rels = {}
    if is_train:
        with open("../mydata/train.txt","r") as train_f:
            for line in train_f.readlines():
                line = line.split('\t')
                trian_rels[line[0].split('-')[-1]] = line[3][3:]

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        data = torch.load(cached_features_file)
        examples = data['examples']
        features = data['features']
        token_time = 0
    #     ##by hn
    #     #demo_fearures = extract_disamb_features_from_examples(args, tokenizer, examples, do_predict=False)
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        candidate_file = 'data/' + SCALE + '/' + MODEL + '/' + stage + '_candidate.json'
        dataset_file = 'data/' + stage + '_data.json'
        examples = read_disamb_instances_from_entity_candidates(dataset_file, candidate_file, redict, trian_rels, is_train=is_train)
        features, token_time = extract_disamb_features_from_examples(args, tokenizer, examples,
                                                                     do_predict=args.do_predict, is_train=is_train)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({'examples': examples, 'features': features}, cached_features_file)

    if output_examples:
        return ListDataset(features), examples, token_time
    else:
        return ListDataset(features), token_time

# def load_and_cache_examples(args, tokenizer, redict, stage, output_examples=False, is_train = True):
#     # Load data features from cache or dataset file
#     input_dir = args.data_dir if args.data_dir else "."
#     SCALE = args.train_file.split('/')[1]
#     MODEL = args.train_file.split('/')[2]
#     os.makedirs('feature_cache/' + SCALE + '/' + MODEL, exist_ok=True)
#     cached_features_file = 'feature_cache/' + SCALE + '/' + MODEL + '/' + stage + '.bin'
#
#     # Init features and dataset from cache if it exists
#     # if os.path.exists(cached_features_file) and not args.overwrite_cache:
#     #     logger.info("Loading features from cached file %s", cached_features_file)
#     #     data = torch.load(cached_features_file)
#     #     examples = data['examples']
#     #     features = data['features']
#     #     token_time=0
#     #     ##by hn
#     #     #demo_fearures = extract_disamb_features_from_examples(args, tokenizer, examples, do_predict=False)
#     # else:
#     logger.info("Creating features from dataset file at %s", input_dir)
#     candidate_file = 'data/' + SCALE + '/' + MODEL + '/' + stage + '_candidate.json'
#     # TODO: hard coded for now
#     '''
#     example_cache = join('feature_cache', '{}_{}.bin'.format(args.dataset,stage))
#     if os.path.exists(example_cache) and not args.overwrite_cache:
#         examples = torch.load(example_cache)
#     else:
#     '''
#     dataset_file = 'data/' + stage + '_data.json'
#     examples = read_disamb_instances_from_entity_candidates(dataset_file, candidate_file, redict)
#     features, token_time = extract_disamb_features_from_examples(args, tokenizer, examples, do_predict=args.do_predict, is_train=is_train)
#
#     logger.info("Saving features into cached file %s", cached_features_file)
#     torch.save({'examples': examples, 'features': features}, cached_features_file)
#
#     if output_examples:
#         return ListDataset(features), examples, token_time
#     else:
#         return ListDataset(features), token_time


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    register_args(parser)
    args = parser.parse_args()
    # choose which gpu to use, -1 for cpu

    SCALE = 'small'
    TYPE = 'bert'
    MODEL ='bert-base-uncased'

    ###
    # if TYPE in ['xlnet','roberta']: 
    #     args.max_seq_length = 192

    # set logger
    global logger
    logger = mylog.log_creater('./log', file_name + '_' + MODEL + '_' + SCALE + '-out')
    logger.warning("\n new process start  \n")
    logger.warning(MODEL + '-' + SCALE)

    args.do_train = True
    args.do_eval = True

    os.makedirs(args.output_dir + '/' + SCALE + '/' + MODEL, exist_ok=True)
    os.makedirs(args.checkpoint_dir + '/' + SCALE + '/' + MODEL, exist_ok=True)

    args.model_type = TYPE

    args.train_file = 'data/' + SCALE + '/' + MODEL + '/train_candidate.json'
    args.predict_file = 'data/' + SCALE + '/' + MODEL + '/valid_candidate.json'

    if SCALE == 'small':
        redict = pickle.load(open('indexes/connect_relation_small.pkl', 'rb'))
    if SCALE == 'large':
        redict = pickle.load(open('indexes/connect_relation_large.pkl', 'rb'))
    if SCALE == 'medium1':
        redict = pickle.load(open('indexes/connect_relation_medium1.pkl', 'rb'))
    if SCALE == 'medium2':
        redict = pickle.load(open('indexes/connect_relation_medium2.pkl', 'rb'))
    
    # delete tokenize cache to calculate tokenize time
    if os.path.exists('feature_cache/' + SCALE + '/' + MODEL):
        temppath = os.listdir('feature_cache/' + SCALE + '/' + MODEL)
        for i in temppath:
            os.remove('feature_cache/' + SCALE + '/' + MODEL + '/' + i)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1

    # # Setup logging
    # logging.basicConfig(
    #     stream=sys.stdout,
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    # )
    # logger.warning(
    #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
    #     args.local_rank,
    #     device,
    #     args.n_gpu,
    #     bool(args.local_rank != -1),
    # )

    # Set seed
    set_seed(args)

    args.model_type = args.model_type.lower()

    # load model for training
    config, tokenizer, model = load_untrained_model(args)

    model.to(args.device)
    logger.warning("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset, token_time = load_and_cache_examples(args, tokenizer, redict, 'train', output_examples=False)
        global_step, tr_loss, train_time = train(args, train_dataset, model, tokenizer, redict)
        # add tokenize time
        train_time += token_time
        logger.warning(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint = args.checkpoint_dir + '/' + SCALE + '/' + MODEL
        logger.warning("Loading checkpoint %s for evaluation", checkpoint)
        logger.warning("Evaluate the following checkpoint: %s", checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model.load_state_dict(torch.load(checkpoint + '/pytorch_model.bin', map_location=args.device))
        model.to(args.device)
        # eval on valid
        logger.warning("********** Start evaluation on valid **********")
        result, valid_time = evaluate(args, model, tokenizer, redict, output_prediction=True)
        result = dict((k, v) for k, v in result.items())
        results.update(result)
        logger.warning("Results: {}".format(results))
        # eval on test, change the eval dataset
        args.predict_file = 'data/' + SCALE + '/' + MODEL + '/test_candidate.json'
        logger.warning("********** Start evaluation on test **********")
        result, test_time = evaluate(args, model, tokenizer, redict, output_prediction=True)
        result = dict((k, v) for k, v in result.items())
        results.update(result)
        logger.warning("Results: {}".format(results))
        logger.warning("train time:{}".format(train_time))
        logger.warning("test time:{}".format(test_time))
    return results


if __name__ == "__main__":
    main()
