import pymysql
import pymysql.cursors

import gc
import json
import re
import os

import pandas as pd

from nltk.tokenize import toktok
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data

fname = 'arxiv.h5'
fname_proc = 'arxiv_proc.h5'
dsname = 'arxiv_data'

if not os.path.exists('arxiv.h5'):
    connection = pymysql.connect(host='localhost',
                                 user='user',
                                 password='*******',
                                 db='arxiv',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    cursor = connection.cursor()
    sql = "select id, title, abstract, categories from documents"
    cursor.execute(sql)
    res = cursor.fetchall()

    df = pd.DataFrame(list(res),
                      columns = ['id', 'title', 'abstract','categories'])

    df.to_hdf(fname, dsname,
              mode='w', format='table',
              data_columns=True)

    # Cleanup resources
    del cursor
    del res
    del df
    gc.collect()

# Process data
toktok = toktok.ToktokTokenizer()
stop_words = stopwords.words('english')

rgc = re.compile("[^a-z-0-9_]")

def preprocess(text, stopwords = stop_words):
    text = rgc.sub(' ', text.lower())
    text = toktok.tokenize(text)
    tokens = [token for token in text if (not token in stopwords) and (not token.isnumeric())]
    return ' '.join(tokens)

for df in pd.read_hdf(fname, dsname, chunksize=100000):
    df['categories'] = df['categories'].apply(lambda x: x.split('|'))
    df['categories'] = df['categories'].apply(lambda x: '|'.join(x))
    df['proc_abstract'] = df['abstract'].apply(preprocess)
    df = df.drop(columns=['abstract'])
    df.to_hdf(fname_proc, dsname,
              mode='a', format='table', append=True,
              data_columns=True,
              min_itemsize={'id': 100, 'title': 500, 'proc_abstract': 10000, 'categories': 200})
