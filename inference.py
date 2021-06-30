#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
from pandas import DataFrame
import pickle
from collections import Counter
from functools import reduce, partial
from multiprocessing import Pool
import numpy as np
from datetime import datetime
from typing import List
from sklearn.preprocessing import LabelEncoder


BEST_MODEL_PATH = "resources/best_model.pickle" #change this line as you wish


with open(BEST_MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

    
def norm_times(row):
    row['Elapsed'] = (row['Close Date'] - row['Created Date']).days
    return row


def preprocess(df: DataFrame, interactions: DataFrame) -> DataFrame:
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Agent'])
    df.Agent = label_encoder.transform(df.Agent)
    
    df.Close_Value = df.groupby('Product').transform(lambda x: x.fillna(x.mean())).Close_Value
    
    label_encoder.fit(df['Product'])
    df.Product = label_encoder.transform(df.Product)
    
    new_df = parallelize_on_rows(df, norm_times)
    X = new_df[['Agent', 'Product', 'Close_Value', 'Elapsed']]
    return X


def inference(df: DataFrame, interactions: DataFrame) -> List[int]:
    """
    path: a DataFrame
    result is the output of function which should be 
    somethe like: [0,1,1,1,0]
    0 -> Lost
    1 -> Won
    --------------------------
    Usage:
    >>> df = pd.read_excel('dataset.xls', index_col=0)
    >>> idf = pd.read_excel('interactions.xlsx')
    >>> inference(df.drop(['Stage'], axis=1), idf)
    [1,
     0,
     0,
     1,
    ...]
    """
    
    df = preprocess(df, interactions)
    result = list(model.predict(df))
    return result
