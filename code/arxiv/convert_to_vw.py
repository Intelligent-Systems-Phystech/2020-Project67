import re
import pandas as pd

fname_stem = 'arxiv_stem.h5'
fname_vw = 'arxiv_abstract_stem.vw'
dsname = 'arxiv_data'

def to_vw(df):
    cat = ' '.join(df['categories'].split('|'))
    return ' |'.join([df['id'],
                      'title ' + df['title'],
                      'abstract ' + df['abstract'],
                      'categories ' + cat + '\n'])

reg = re.compile(':|;')

f = open(fname_vw, 'w')
for df in pd.read_hdf(fname_stem, dsname, chunksize=50000):
    df = df.apply(to_vw, axis=1)
    lines = ''.join([re.sub(reg, '', line) for line in df.values.tolist()])
    f.write(lines)
f.close()
