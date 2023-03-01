import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np
from transformers import LukeModel, GPT2ForSequenceClassification, LukeConfig, GPT2Config,AutoModel,AutoConfig

class Bertforclass(nn.Module):
    def __init__(self, args):
        super(Bertforclass, self).__init__()
        self.args = args
        MODEL=args.model
        label=args.label
        config=AutoConfig.from_pretrained("pretrain/"+MODEL)
        self.bert=AutoModel.from_pretrained("pretrain/"+MODEL)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, label)
    
    def forward(self, input_ids,attention_mask,labels=None):
        x=self.bert(input_ids=input_ids,attention_mask=attention_mask).pooler_output
        x=self.dropout(x)
        x=self.classifier(x)
        loss=None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(x.view(-1, self.args.label), labels.view(-1))
        return loss,x

class Lukeforclass(nn.Module):
    def __init__(self, args):
        super(Lukeforclass, self).__init__()
        self.args = args
        MODEL=args.model
        label=args.label
        config=LukeConfig.from_pretrained("../pretrain/"+MODEL)
        self.bert=LukeModel.from_pretrained("../pretrain/"+MODEL)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, label)

    def forward(self, input_ids,attention_mask,labels=None):
        x=self.bert(input_ids=input_ids,attention_mask=attention_mask).pooler_output
        x=self.dropout(x)
        x=self.classifier(x)
        loss=None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(x.view(-1, self.args.label), labels.view(-1))
        return loss,x
        
class GPT2forclass(nn.Module):
    def __init__(self, args):
        super(GPT2forclass, self).__init__()
        self.args = args
        MODEL=args.model
        label=args.label
        self.bert=GPT2ForSequenceClassification.from_pretrained("../pretrain/"+MODEL,num_labels=label)
        self.bert.config.pad_token_id = self.bert.config.eos_token_id
    
    def forward(self, input_ids,attention_mask,labels=None):
        x=self.bert(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        return x.loss,x.logits