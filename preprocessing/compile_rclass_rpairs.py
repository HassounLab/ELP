import pandas as pd
from collections import defaultdict
import os
import re
import numpy as np
import pickle
rclass_file = '%s/kegg/Ligand_April_20_2020/rclass/rclass' % os.environ['DATAPATH']
cofactors_file = '%s/kegg/cofactor.lst' % os.environ['DATAPATH']


with open(cofactors_file, encoding='utf-8-sig') as f:
    cofactors = f.read().splitlines()
cofactors = set(['C' + '0' * (5 - len(x)) + x for x in cofactors])

rclass_to_rpairs = {}
rpair_to_rclass = defaultdict(list)
df = pd.DataFrame(columns=['rclass', 'num_rpairs', 'num_rpairs_cofac'])
with open(rclass_file) as f:
    rc = None
    rpair_lines = False
    for line in f:
        if line.startswith('ENTRY'):
            rc = line.split()[1]
            assert rc.startswith('RC') and len(rc) == 7
            rclass_to_rpairs[rc] = []
            num_cofac = 0
        elif line.startswith('RPAIR'):
            rpair_lines = True
            line = ' '.join(line.split()[1:])
        if rpair_lines:
            fstwd = line.split()[0]
            if fstwd.isupper() and fstwd.isalpha() or fstwd == '///':
                rpair_lines = False
                df.loc[len(df)] = [rc, len(rclass_to_rpairs[rc]), num_cofac]
                rc = None
                continue
            for rpair in line.split():
                c1, c2 = rpair.split('_')
                assert c1.startswith('C') and len(c1) == 6
                assert c2.startswith('C') and len(c2) == 6
                assert rc is not None
                if c2 < c1:
                    c1, c2 = c2, c1
                rclass_to_rpairs[rc].append((c1, c2))
                if c1 in cofactors or c2 in cofactors:
                    num_cofac += 1
                else:
                    rpair_to_rclass[(c1, c2)].append(rc)
print('Number of rclasses', len(rclass_to_rpairs))
print('Avg rpairs per rclass', np.mean([len(x) for x in rclass_to_rpairs.values()]))
print('Avg num cofacs per rclass', df['num_rpairs_cofac'].mean())
print('Avg rclass per rpairs', np.mean([len(x) for x in rpair_to_rclass.values()]))
print('Num rpairs with > 1 rclasses', sum([len(x) > 1 for x in rpair_to_rclass.values()]))
for rp, x in rpair_to_rclass.items():
    if len(x) > 1:
        print(rp, x)
with open('%s/kegg/rclass_to_edges.pkl' % os.environ['DATAPATH'], 'rb') as f:
    rclass_to_edges = pickle.load(f)

for i in df.index:
    rc = df.loc[i, 'rclass']
    if rc not in rclass_to_edges:
        continue
    df.loc[i, 'num_edges'] = len(rclass_to_edges[rc])

print(df)
df['num_rpairs'] = df['num_rpairs'].astype(int)
df['num_rpairs_cofac'] = df['num_rpairs_cofac'].astype(int)
print(df[['num_rpairs', 'num_rpairs_cofac', 'num_edges']].describe())

rclass_stats = '%s/kegg/rclass_stats.csv' % os.environ['DATAPATH']
df.to_csv(rclass_stats, index=False)
print('Saved to', rclass_stats)



import IPython
IPython.embed()
