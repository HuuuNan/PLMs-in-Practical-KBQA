import torch
import torch.nn as nn
from transformers import AutoModel,AutoConfig,LukeConfig,LukeModel,GPT2Config,GPT2ForTokenClassification

class Ner(nn.Module):
    def __init__(self, model=None, vocab_size=None, device='cpu'):
        super().__init__()
        config=AutoConfig.from_pretrained(model)
        self.bert = AutoModel.from_pretrained(model)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, vocab_size)
        self.device = device

    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)
        
        x = self.bert(x).last_hidden_state
        x= self.dropout(x)
        x=self.classifier(x)
        y_hat = x.argmax(-1)
        
        return x, y, y_hat

class LukeNer(nn.Module):
    def __init__(self, model=None, vocab_size=None, device='cpu'):
        super().__init__()
        config=LukeConfig.from_pretrained(model)
        self.bert = LukeModel.from_pretrained(model)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, vocab_size)
        self.device = device

    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)
        
        x = self.bert(x).last_hidden_state
        x= self.dropout(x)
        x=self.classifier(x)
        y_hat = x.argmax(-1)
        
        return x, y, y_hat
        
class GPT2Ner(nn.Module):
    def __init__(self, model=None, vocab_size=None, device='cpu'):
        super().__init__()
        config=GPT2Config.from_pretrained(model)
        self.bert = GPT2ForTokenClassification(config).from_pretrained(model,num_labels=3)
        self.bert.config.id2label = self.bert.config.eos_token_id
        self.device = device

    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)
        
        x = self.bert(x).logits
        y_hat = x.argmax(-1)
        
        return x, y, y_hat