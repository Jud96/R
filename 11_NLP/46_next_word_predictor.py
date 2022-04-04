#!/usr/bin/env python
# coding: utf-8

# In[13]:



import torch
import string
from transformers import BertTokenizer, BertForMaskedLM
#from pynput import keyboard


# ## Load Model and Tokenizer
# First of all Load BERT model. Here I have used uncased bert that means it is case-insensitive and point to remember is that model will be in eval mode because we are not training model but using pretrained model. Also Load Bert tokenizer that tokenizes the text.

# In[8]:


def load_model(model_name):
    try:
        if model_name.lower() == "bert":
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
            return bert_tokenizer,bert_model
    except Exception as e:
        pass


# ## Add mask at the end
# As we want to predict the last word in given text so it is required to add a mask token at the end of input text becasue BERT requires input to be preprocessed in this way.

# In[9]:


def get_prediction_eos(input_text):
      try:
        input_text += ' <mask>'
        res = get_all_predictions(input_text, top_clean=int(top_k))
        return res
      except Exception as error:
        pass


# ## Encode and Decode
# In this step input text is encoded with bert tokenizer . Here I have used add_special_tokens = True because I want to encode out-of-vocabulary words aka UNK with special token that BERT uses. Then, when tokenizer encodes the input text it returns input_ids . After that get mask index (mask_idx) that is the place where mask has been added.

# In[10]:


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
              tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])




def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

        input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
        mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


# ## Wrapper function for encoder and decoder
# Both of the above mentioned functions have been wrapped in the following function.

# In[11]:


def get_all_predictions(text_sentence, top_clean=5):
    # ========================= BERT =================================
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    return {'bert': bert}


# ## Interface
# Here is some extra code that has been used for developing streamlit webapp interface

# In[14]:


def get_all_predictions(text_sentence, top_clean=5):
    # ========================= BERT =================================
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    return {'bert': bert}

model_name = 'BERT'
bert_tokenizer, bert_model  = load_model(model_name)
## return string results 
top_k = 3 #some times it is possible to have less words
def str_return(input_text ):   
  
        
    
        
      
         
        #input_text = "how many people"
        #click outside box of input text to get result
        res = get_prediction_eos(input_text)
    
        answer = []
        #print(res['bert'].split("\n"))
        for i in res['bert'].split("\n"):
              answer.append(i)
        answer_as_string = " ".join(answer)
        return  answer_as_string
