import sys
import re
import pandas as pd
import IPython
import os
from io import StringIO

def replace_comma(g):
    return g.group(0).replace(',', '')

def parse_df(fn):
    with open(fn) as f:
        data = f.read()
        data = re.sub(r'\(.*?\)', replace_comma, data)
        data = StringIO(data)
        df = pd.read_csv(data)
    return df

while True:
    f = input('file: ')
    if not os.path.exists(f):
        print(f, 'does not exist')
    else:
        df = parse_df(f)
        IPython.embed()
