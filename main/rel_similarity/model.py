import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import unicodedata
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
import torch.nn.functional as F
import random
from torch.nn import CosineSimilarity
from collections import Counter
from field import *
from transformers import AutoModel,GPT2ForSequenceClassification

from torch.nn import CrossEntropyLoss


class BertCharEmbedding(nn.Module):
    def __init__(self, model, requires_grad=True):
        super(BertCharEmbedding, self).__init__()
        self.model_name = model
        if model == 'gpt2':
            self.bert=GPT2ForSequenceClassification.from_pretrained("../pretrain/"+model)
            self.bert.config.pad_token_id = self.bert.config.eos_token_id
        else:
            self.bert=AutoModel.from_pretrained("../pretrain/"+model)
        self.requires_grad = requires_grad
    
    def forward(self, subwords, bert_mask):
        if self.model_name == 'gpt2':
            bert= self.bert(subwords, attention_mask=bert_mask, return_dict=True).logits
        else: 
            bert= self.bert(subwords, attention_mask=bert_mask, return_dict=True).last_hidden_state
        return bert

class Bert_Comparing(nn.Module):
    def __init__(self, data):
        super(Bert_Comparing, self).__init__()

        self.question_bert_embedding = BertCharEmbedding(data.model, data.requires_grad)
        self.path_bert_embedding = BertCharEmbedding(data.model, data.requires_grad)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.model_name = data.model
        self.args = data
        self.similarity = CosineSimilarity(dim=1)
    
    def question_encoder(self, input_idxs, bert_mask):
        bert_outs = self.question_bert_embedding(input_idxs, bert_mask)
        if self.model_name == 'gpt2':
            # return bert_outs[:,-1]
            return bert_outs
        return bert_outs[:, 0]
    
    def path_encoder(self, input_idxs, bert_mask):
        # bert_outs = self.question_bert_embedding(input_idxs, bert_mask)
        bert_outs = self.path_bert_embedding(input_idxs, bert_mask)
        if self.model_name == 'gpt2':
            return bert_outs
        return bert_outs[:, 0]
        
    def forward(self, questions, pos, negs):
        '''
        questions: batch_size, max_seq_len

        pos_input_idxs: batch_size, max_seq_len
        pos_bert_lens: batch_size, max_seq_len
        pos_bert_mask: batch_size, max_seq_len

        neg_input_idxs: neg_size, batch_size, max_seq_len
        neg_bert_lens: neg_size, batch_size, max_seq_len
        neg_bert_mask: neg_size, batch_size, max_seq_len
        '''
        
        (q_input_idxs, q_bert_mask) = questions
        q_bs, q_dim= q_input_idxs.shape

        (pos_input_idxs, pos_bert_mask) = pos
        (neg_input_idxs, neg_bert_mask) = negs
        neg_size, batch_size, _ = neg_input_idxs.shape

        # q_input_idxs = q_input_idxs.unsqueeze(0).expand(neg_size+1, q_bs, q_dim)
        # q_bert_mask = q_bert_mask.unsqueeze(0).expand(neg_size+1, q_bs, q_dim)

        # pos_input_idxs = pos_input_idxs.unsqueeze(0)
        # pos_bert_mask = pos_bert_mask.unsqueeze(0)
        # seg_input_idxs = torch.cat((pos_input_idxs, neg_input_idxs),0)
        # seg_bert_mask = torch.cat((q_bert_mask, neg_bert_mask),0)

        # input_idxs = torch.cat((q_input_idxs,seg_input_idxs),2)
        # input_bert_mask = torch.cat((q_bert_mask,seg_bert_mask),2)
        # q_encoding = self.question_encoder(input_idxs, input_bert_mask) # (batch_size, hidden_dim)
        # logits = self.classifier(q_encoding)
        # logits = logits.squeeze(-1).transpose(0,1)

        # labels= torch.zeros(batch_size)
        # loss_fct = CrossEntropyLoss()

        # loss = loss_fct(logits, labels)
        # return loss

        q_encoding = self.question_encoder(q_input_idxs, q_bert_mask) # (batch_size, hidden_dim)

        pos_encoding = self.path_encoder(pos_input_idxs, pos_bert_mask)

        neg_input_idxs = neg_input_idxs.reshape(neg_size*batch_size, -1) # (neg_size*batch_size, max_seq_len)
        neg_bert_mask = neg_bert_mask.reshape(neg_size*batch_size, -1) # (neg_size*batch_size, max_seq_len)

        neg_encoding = self.path_encoder(neg_input_idxs, neg_bert_mask) # (neg_size*batch_size, hidden_dim)
        # p_encoding = p_encoding.reshape(neg_size, batch_size, -1) # (neg_size, batch_size, hidden_dim)
        
        q_encoding_expand = q_encoding.unsqueeze(0).expand(neg_size, batch_size, q_encoding.shape[-1]).reshape(neg_size*batch_size, -1) # (neg_size*batch_size, hidden_dim)

        pos_score = self.similarity(q_encoding, pos_encoding)
        pos_score = pos_score.unsqueeze(1) # (batch_size, 1)
        neg_score = self.similarity(q_encoding_expand, neg_encoding)
        neg_score = neg_score.reshape(neg_size,-1).transpose(0,1) # (batch_size, neg_size)

        return (pos_score, neg_score)
    
    @torch.no_grad()
    def cal_score(self, question, cands, pos=None):
        '''
        one question, several candidate paths
        question: (max_seq_len), (max_seq_len), (max_seq_len)
        cands: (batch_size, max_seq_len), (batch_size, max_seq_len), (batch_size, max_seq_len)
        '''
        
        question = (t.unsqueeze(0) for t in question)

        if self.args.no_cuda == False:
            question = (t.cuda() for t in question)

        (q_input_idxs, q_bert_mask) = question
        
        
        q_encoding = self.question_encoder(q_input_idxs, q_bert_mask) # (batch_size=1, hidden_dim)
        
        if pos:
            pos = (t.unsqueeze(0) for t in pos)
            if self.args.no_cuda == False:
                pos = (t.cuda() for t in pos)
            
            (pos_input_idxs, pos_bert_mask) = pos
            pos_encoding = self.path_encoder(pos_input_idxs, pos_bert_mask) # (batch_size=1, hidden_dim)
            pos_score = self.similarity(q_encoding, pos_encoding) # (batch_size=1) 

        all_scores = []

        for (batch_input_idxs, batch_bert_mask) in cands:
            if self.args.no_cuda ==False:
                batch_input_idxs, batch_bert_mask = batch_input_idxs.cuda(), batch_bert_mask.cuda()
            path_encoding = self.path_encoder(batch_input_idxs, batch_bert_mask) #(batch_size, hidden_dim)
            q_encoding_expand = q_encoding.expand_as(path_encoding)
            scores = self.similarity(q_encoding_expand, path_encoding) # (batch_size)
            for score in scores:
                all_scores.append(score)
        all_scores = torch.Tensor(all_scores)

        if pos:
            return pos_score.cpu(), all_scores.cpu()
        else:
            return all_scores.cpu()
