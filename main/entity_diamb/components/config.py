"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import random

import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoConfig,
    RobertaConfig,
    RobertaTokenizer,
)

from .model import CandidateRanking,DistilbertRanking,XlnetRanking,GPT2Ranking,KEPLERRanking


from .models.BertRanker import BertForCandidateRanking
from .models.RobertaRanker import RobertaForCandidateRanking
MODEL_TYPE_DICT = {
    'bert': BertForCandidateRanking,
    'roberta': RobertaForCandidateRanking,
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def register_args(parser):
    # Required parameters
    parser.add_argument("--patience", default=3)
    parser.add_argument(
        "--dataset",
        default='SQ',
        type=str,
        help="dataset to operate on",
    )
    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        help="Model type",
    )
    parser.add_argument(
        "--model_name_or_path",
        default='pretrain/bert-base-uncased',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default='result',
        type=str,
        help="The output directory where the predictions will be written.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default='checkpoint',
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default='data',
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default='data/bert-base-uncased/train_candidate.json',
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default='data/bert-base-uncased/valid_candidate.json',
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="./hfcache",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--max_seq_length",
        default=96,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", default=False, help="Whether to do prediction.")
    parser.add_argument(
        "--evaluate_during_training", default=True, help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", default=True, help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1000, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Linear warmup over warmup ratio.")
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    parser.add_argument("--logging_steps", type=int, default=10000, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=8000, help="Eval every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--disable_tqdm", action="store_true", help="Disable tqdm bar"
    )
    parser.add_argument("--num_contrast_sample", type=int, default=20, help="number of samples in a batch.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", default=True, help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    # train curriculum
    parser.add_argument("--training_curriculum", default="random",type=str, choices=["random", "bootstrap", "mixbootstrap"])
    parser.add_argument("--bootstrapping_start", default=None, type=int, help="when to start bootstrapping sampling")
    parser.add_argument("--bootstrapping_ticks", default=None, type=str, help="when to update scores for bootstrapping in addition to the startpoint")

    # textualizing choices
    parser.add_argument("--linear_method", default="vanilla",type=str, choices=["vanilla", "naive_text", "reduct_text"])

def validate_args(args):
    # validate before loading data
    if args.training_curriculum == "random":
        args.bootstrapping_update_epochs = []
    else:
        assert args.bootstrapping_start is not None
        assert args.bootstrapping_start > 0

        if args.bootstrapping_ticks is None:
            bootstrapping_update_epochs = [args.bootstrapping_start]
        else:
            additional_update_epochs = [int(x) for x in args.bootstrapping_ticks.split(',')]
            bootstrapping_update_epochs = [args.bootstrapping_start] + additional_update_epochs
        args.bootstrapping_update_epochs = bootstrapping_update_epochs

def load_untrained_model(args):
    MODEL=args.train_file.split('/')[2]
    args.model_type = args.model_type.lower()
    
    if MODEL!='KEPLER':
        config = AutoConfig.from_pretrained('../pretrain/'+MODEL)
        tokenizer = AutoTokenizer.from_pretrained('../pretrain/'+MODEL)
    else:
        config=RobertaConfig.from_pretrained(args.model_name_or_path)
        tokenizer=RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model=KEPLERRanking('../pretrain/'+MODEL,config)
        
    if args.model_type=='distilbert':
        model=DistilbertRanking('../pretrain/'+MODEL,config)
    if args.model_type=='xlnet':
        model=XlnetRanking('../pretrain/'+MODEL,config)
    if args.model_type=='gpt2':
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.sep_token=tokenizer.eos_token
        model=GPT2Ranking('../pretrain/'+MODEL,config)
    
    if args.model_type not in ['distilbert','xlnet','gpt2'] and MODEL!='KEPLER':
        model=CandidateRanking('../pretrain/'+MODEL,config)
    #     model = BertForCandidateRanking.from_pretrained(
    #     '../pretrain/'+MODEL,
    #     config=config,
    # )

    return config, tokenizer, model