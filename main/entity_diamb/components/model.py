"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from logging import log
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from .models.model_utils import get_inf_mask
from transformers import AutoModel,AutoModelForSequenceClassification,DistilBertForSequenceClassification,XLNetForSequenceClassification,XLNetModel,GPT2ForSequenceClassification,RobertaModel
from transformers.modeling_utils import SequenceSummary

'''
class CandidateRanking(nn.Module):
    def __init__(self, config):
        super(CandidateRanking, self).__init__()

        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        #self.init_weights()

    # for training return loss, [batch_size * num_sample]
    # for testing, batch size have to be 1
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sample_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        # for training, input is batch_size * sample_size * L
        # for testing, it is batch_size * L
        if labels is not None:
            batch_size = input_ids.size(0)
            sample_size = input_ids.size(1)
            # flatten first two dim
            input_ids = input_ids.view((batch_size * sample_size,-1))
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view((batch_size * sample_size,-1))
            attention_mask = attention_mask.view((batch_size * sample_size,-1))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
    
        loss = None
        
        if labels is not None:
            # reshape logits
            logits = logits.view((batch_size, sample_size))
            logits = logits + get_inf_mask(sample_mask)
            # apply infmask
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        else:
            logits = logits.squeeze(1)
        
        output = (logits,) + outputs[2:]
        
        #return ((loss,) + output) if loss is not None else output
        return ((loss,) + output)
'''
  
class CandidateRanking(nn.Module):
    def __init__(self, model_path,config):
        super(CandidateRanking, self).__init__()

        self.bert = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        #self.init_weights()

    # for training return loss, [batch_size * num_sample]
    # for testing, batch size have to be 1
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sample_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert labels is not None
        # for training, input is batch_size * sample_size * L
        # for testing, it is batch_size * L
        batch_size = input_ids.size(0)
        sample_size = input_ids.size(1)
        # flatten first two dim
        input_ids = input_ids.view((batch_size * sample_size,-1))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((batch_size * sample_size,-1))
        attention_mask = attention_mask.view((batch_size * sample_size,-1))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        
        # reshape logits
        logits = logits.view((batch_size, sample_size))
        logits = logits + get_inf_mask(sample_mask)
        # apply infmask
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels.view(-1))
        
        return ((loss,) + (logits,))

class DistilbertRanking(nn.Module):
    def __init__(self, model_path,config):
        super(DistilbertRanking, self).__init__()

        self.bert = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=768)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.dim, 1)

        #self.init_weights()

    # for training return loss, [batch_size * num_sample]
    # for testing, batch size have to be 1
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sample_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        # for training, input is batch_size * sample_size * L
        # for testing, it is batch_size * L
        if labels is not None:
            batch_size = input_ids.size(0)
            sample_size = input_ids.size(1)
            # flatten first two dim
            input_ids = input_ids.view((batch_size * sample_size,-1))
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view((batch_size * sample_size,-1))
            attention_mask = attention_mask.view((batch_size * sample_size,-1))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.logits

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
    
        loss = None
        
        if labels is not None:
            # reshape logits
            logits = logits.view((batch_size, sample_size))
            logits = logits + get_inf_mask(sample_mask)
            # apply infmask
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        else:
            logits = logits.squeeze(1)
        
        output = (logits,) + outputs[2:]
        
        #return ((loss,) + output) if loss is not None else output
        return ((loss,) + output)

class XlnetRanking(nn.Module):
    def __init__(self,model_path,config):
        super(XlnetRanking, self).__init__()

        self.bert=XLNetModel.from_pretrained(model_path)
        self.sequence_summary = SequenceSummary(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, 1)

        #self.init_weights()

    # for training return loss, [batch_size * num_sample]
    # for testing, batch size have to be 1
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sample_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        # for training, input is batch_size * sample_size * L
        # for testing, it is batch_size * L
        if labels is not None:
            batch_size = input_ids.size(0)
            sample_size = input_ids.size(1)
            # flatten first two dim
            input_ids = input_ids.view((batch_size * sample_size,-1))
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view((batch_size * sample_size,-1))
            attention_mask = attention_mask.view((batch_size * sample_size,-1))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.last_hidden_state
        pooled_output = self.sequence_summary(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
    
        loss = None
        
        if labels is not None:
            # reshape logits
            logits = logits.view((batch_size, sample_size))
            logits = logits + get_inf_mask(sample_mask)
            # apply infmask
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        else:
            logits = logits.squeeze(1)
        
        output = (logits,) + outputs[2:]
        
        #return ((loss,) + output) if loss is not None else output
        return ((loss,) + output)

class GPT2Ranking(nn.Module):
    def __init__(self, model_path,config):
        super(GPT2Ranking, self).__init__()

        self.bert = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=768)
        self.bert.config.pad_token_id = self.bert.config.eos_token_id
        self.dropout = nn.Dropout(config.embd_pdrop)
        self.classifier = nn.Linear(config.n_embd, 1)

        #self.init_weights()

    # for training return loss, [batch_size * num_sample]
    # for testing, batch size have to be 1
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sample_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        # for training, input is batch_size * sample_size * L
        # for testing, it is batch_size * L
        if labels is not None:
            batch_size = input_ids.size(0)
            sample_size = input_ids.size(1)
            # flatten first two dim
            input_ids = input_ids.view((batch_size * sample_size,-1))
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view((batch_size * sample_size,-1))
            attention_mask = attention_mask.view((batch_size * sample_size,-1))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.logits

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
    
        loss = None
        
        if labels is not None:
            # reshape logits
            logits = logits.view((batch_size, sample_size))
            logits = logits + get_inf_mask(sample_mask)
            # apply infmask
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        else:
            logits = logits.squeeze(1)
        
        output = (logits,) + outputs[2:]
        
        #return ((loss,) + output) if loss is not None else output
        return ((loss,) + output)

class KEPLERRanking(nn.Module):
    def __init__(self, model_path,config):
        super(KEPLERRanking, self).__init__()

        self.bert = RobertaModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        #self.init_weights()

    # for training return loss, [batch_size * num_sample]
    # for testing, batch size have to be 1
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sample_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert labels is not None
        # for training, input is batch_size * sample_size * L
        # for testing, it is batch_size * L
        batch_size = input_ids.size(0)
        sample_size = input_ids.size(1)
        # flatten first two dim
        input_ids = input_ids.view((batch_size * sample_size,-1))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((batch_size * sample_size,-1))
        attention_mask = attention_mask.view((batch_size * sample_size,-1))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        
        # reshape logits
        logits = logits.view((batch_size, sample_size))
        logits = logits + get_inf_mask(sample_mask)
        # apply infmask
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels.view(-1))
        
        return ((loss,) + (logits,))

'''
class XlnetRanking(nn.Module):
    def __init__(self,model_path,config):
        super(XlnetRanking, self).__init__()

        self.bert=XLNetForSequenceClassification.from_pretrained(model_path,num_labels=768)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, 1)

        #self.init_weights()

    # for training return loss, [batch_size * num_sample]
    # for testing, batch size have to be 1
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sample_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        # for training, input is batch_size * sample_size * L
        # for testing, it is batch_size * L
        if labels is not None:
            batch_size = input_ids.size(0)
            sample_size = input_ids.size(1)
            # flatten first two dim
            input_ids = input_ids.view((batch_size * sample_size,-1))
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view((batch_size * sample_size,-1))
            attention_mask = attention_mask.view((batch_size * sample_size,-1))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.logits
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
    
        loss = None
        
        if labels is not None:
            # reshape logits
            logits = logits.view((batch_size, sample_size))
            logits = logits + get_inf_mask(sample_mask)
            # apply infmask
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        else:
            logits = logits.squeeze(1)
        
        output = (logits,) + outputs[2:]
        
        #return ((loss,) + output) if loss is not None else output
        return ((loss,) + output)
'''