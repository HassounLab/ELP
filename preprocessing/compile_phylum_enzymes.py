import pandas as pd 
import os
import networkx as nx
import pickle
import re
from collections import defaultdict
LIGAND = '%s/kegg/Ligand_April_20_2020/' % os.environ['DATAPATH']

def compile_organism_edges():
    df = pd.read_csv('%s/kegg/organism_enzymes.csv' % os.environ['DATAPATH'])
    df.columns = [c.lstrip().replace(' ', '_') for c in df.columns]
    phylums = ['Proteobacteria', 'Firmicutes', 'Bacteroidetes', 'Actinobacteria']
    df = df[df['phylum'].isin(phylums)]
    df['organism'] = df['organism'].str.lstrip()
    
    print('# organism per phylum')
    print(df.groupby('phylum')['organism'].size())    

    df['enzymes'] = df['enzymes'].apply(
            lambda ezs: ' '.join([e for e in ezs.split() if sum(1 for x in e if x == '-') <= 1]))
    phylum_to_enzymes = {}
    for phylum, df_p in df.groupby('phylum'):
        enzymes = set([])
        for _, ezs in df_p['enzymes'].iteritems():
            for e in ezs.split():
                if sum(1 for x in e if x == '-') > 1:
                    continue
                enzymes.add(e)
        phylum_to_enzymes[phylum] = enzymes

    print('Number of enzymes associated with each phylum')
    for k, v in phylum_to_enzymes.items():
        print(k, len(v))


    phylum_to_unique_enzymes = {}

    for p in phylums:
        p1, p2, p3 = [po for po in phylums if po != p]
        phylum_to_unique_enzymes[p] = phylum_to_enzymes[p] - \
                phylum_to_enzymes[p1] - phylum_to_enzymes[p2] - phylum_to_enzymes[p3]  



    print('Number of unique enzymes associated with each phylum')
    for k, v in phylum_to_unique_enzymes.items():
        print(k, len(v))


    df = pd.DataFrame(index=phylums, columns=['enzymes', 'n_enzymes', 'unique_enzymes', 'n_unique_enzymes'])
    for p in phylums:
        df.loc[p] = [' '.join(phylum_to_enzymes[p]), len(phylum_to_enzymes[p]), 
                     ' '.join(phylum_to_unique_enzymes[p]), len(phylum_to_unique_enzymes[p])]

    df = df.reset_index().rename(columns={'index': 'phylum'})

    df.to_csv('%s/kegg/phylum_enzymes.csv' % os.environ['DATAPATH'], index=False)


def compile_enzymes_edges():

    ec_reaction_list = '%s/enzyme/links/enzyme_reaction.list' % LIGAND
    
    ec_to_rn = defaultdict(list)
    with open(ec_reaction_list) as f:
        for line in f:
            ec, rn = line.split()
            ec, rn = ec[3:], rn[3:]
            assert sum(1 for x in ec if x == '.') == 3, ec
            assert rn.startswith('R') and len(rn) == 6, rn
            ec_to_rn[ec].append(rn)
    df = pd.DataFrame([[ec, '.'.join(ec.split('.')[:-1]), ' '.join(rns), len(rns)] \
                            for ec, rns in ec_to_rn.items()], 
                      columns=['ec', 'ec_sup', 'reactions', 'n_reactions'])
    df_phylum = pd.read_csv('%s/kegg/phylum_enzymes.csv' % os.environ['DATAPATH'])
    for p in df_phylum['phylum'].values:
        df[p] = False
    df = df.set_index('ec')
    for _, row in df_phylum.iterrows():
        p, enzymes = row['phylum'], row['enzymes']
        for e in enzymes.split():
            if '-' in e:
                where = df.index[df['ec_sup'] == '.'.join(e.split('.')[:-1])]
                if len(where) == 0:
                    print('Unabled to find', e)
                df.loc[where, p] = True
            else:
                if e not in df.index:
                    print('Unabled to find', e)
                else:
                    df.loc[e, p] = True
    phylums = df_phylum['phylum'].values
    for i, row in df.iterrows():
        if sum(row[phylums]) == 1:
            phylum = [p for p in phylums if row[p]][0]
            df.loc[i, 'unique_phylum'] = phylum
    for p in phylums:
        print(df[p].value_counts())
    print(df['unique_phylum'].value_counts(dropna=False))
    dfpath = '%s/kegg/ec_phylum_reaction.csv' % os.environ['DATAPATH']
    df = df.reset_index()
    df.to_csv(dfpath, index=False)
    print('Saved to', dfpath)
    print('Number of reactions associated with each phylum')
    phylum_to_reactions = {}
    for p in phylums:
        reactions = set([])
        for _, rns in df[df[p]]['reactions'].iteritems():
            reactions |= set(rns.split())
        print(p, len(reactions))
        phylum_to_reactions[p] = reactions
    reaction_lst = '%s/reaction/reaction.lst' % LIGAND
    rn_to_edges = {}
    G = nx.read_edgelist('%s/kegg/kegg_2020_consolidated.edgelist' % os.environ['DATAPATH'])
    with open(reaction_lst) as f:
        compound_re = re.compile("(C\d+)")
        for line in f:
            rn = line[:6]
            rn_to_edges[rn] = set([])
            assert rn.startswith('R') and rn[1:].isdigit()
            lhs, rhs = line.split("<=>")
            reactants = compound_re.findall(lhs)
            products  = compound_re.findall(rhs)
            for r in reactants:
                for p in products:
                    c1, c2 = min(r, p), max(r, p) 
                    if c1 != c2 and G.has_edge(c1, c2):
                        rn_to_edges[rn].add((c1, c2))
    print(nx.info(G))
    print('Number of edges associated with each phylum')
    phylum_to_edges = {}
    for phylum in phylums:
        phylum_to_edges[phylum] = set([])
        for rn in phylum_to_reactions[phylum]:
            for (r, p) in rn_to_edges[rn]:
                phylum_to_edges[phylum].add((r, p))
        print('%s %d (%.2f)' \
              % (phylum, len(phylum_to_edges[phylum]), 
                 len(phylum_to_edges[phylum]) / G.number_of_edges())) 
    ptefile = '%s/kegg/phylum_to_edges.pkl' % os.environ['DATAPATH']
    with open(ptefile, 'wb') as f:
        pickle.dump(phylum_to_edges, f)
    print('Dumped to', ptefile)
compile_enzymes_edges()
