import pandas as pd

from nltk.stem import snowball

fname_proc = 'arxiv_proc.h5'
fname_stem = 'arxiv_stem.h5'
dsname = 'arxiv_data'

stemmer = snowball.EnglishStemmer()

def stem_str(s):
    return ' '.join([stemmer.stem(w) for w in s.split()])

for df in pd.read_hdf(fname_proc, dsname, chunksize=100000):
    df['title'] = df['title'].apply(stem_str)
    df['abstract'] = df['proc_abstract'].apply(stem_str)
    df = df.drop(columns=['proc_abstract'])
    df.to_hdf(fname_stem, dsname,
              mode='a', format='table', append=True,
              data_columns=True,
              min_itemsize={'id': 100, 'title': 500, 'abstract': 10000, 'categories': 200})
