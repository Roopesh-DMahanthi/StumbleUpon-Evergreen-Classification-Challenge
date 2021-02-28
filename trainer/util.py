from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from transformers import DistilBertTokenizer
distil_bert = 'distilbert-base-uncased'

def filter_stopwords(sentence):
  from nltk.corpus import stopwords 
  from nltk.tokenize import word_tokenize 

  stop_words = set(stopwords.words('english')) 

  word_tokens = word_tokenize(sentence) 

  filtered_sentence = [w for w in word_tokens if not(w in stop_words) and w !='``' ]  

  return (' ').join(filtered_sentence)

def load_data():
    """Loads data into preprocessed (train_x, train_y, eval_y, eval_y)
    dataframes.
    Returns:
      A tuple (train_x, train_y, eval_x, eval_y), where train_x and eval_x are
      Pandas dataframes with features for training and train_y and eval_y are
      numpy arrays with the corresponding labels.
    """
    df_train=pd.read_csv('https://github.com/Roopesh-DMahanthi/StumbleUpon-Evergreen-Classification-Challenge/raw/main/train.tsv',sep='\t',usecols=['urlid','boilerplate','label'])
    df_train['boilerplate'].replace(to_replace=r'"title":', value="",inplace=True,regex=True)
    df_train['boilerplate'].replace(to_replace=r'"body":', value="",inplace=True,regex=True)
    df_train['boilerplate'].replace(to_replace=r'"url":',value="",inplace=True,regex=True)

    df_train['boilerplate'].replace(to_replace=r'{|}',value="",inplace=True,regex=True)
    df_train['boilerplate']=df_train['boilerplate'].str.lower()
    df_train['boilerplate']=df_train['boilerplate'].apply(filter_stopwords)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    for train_index, test_index in split.split(df_train,df_train["label"]):
        strat_train_set = df_train.loc[train_index]
        strat_val_set = df_train.loc[test_index]
    strat_train_set = strat_train_set.reset_index(drop=True)
    strat_val_set = strat_val_set.reset_index(drop=True)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(strat_train_set['boilerplate'].to_list(), truncation=True, padding=True)
    val_encodings = tokenizer(strat_val_set['boilerplate'].to_list(), truncation=True, padding=True)


    return dict(train_encodings),strat_train_set['label'],dict(val_encodings),strat_val_set['label']