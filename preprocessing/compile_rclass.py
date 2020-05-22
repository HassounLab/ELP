from argparse import ArgumentParser
from collections import defaultdict
import re
import pickle
import os
import numpy as np
LIGAND = '%s/kegg/Ligand_April_20_2020/' % os.environ['DATAPATH']

def compile_rclass(reaction_lst, rclass_reaction_list, output_file,
                   cofactors_file=None):
    if cofactors_file is not None:
        print('Filtering for cofactors')
        with open(cofactors_file, encoding='utf-8-sig') as f:
            cofactors = f.read().splitlines()
        cofactors = set(['C' + '0' * (5 - len(x)) + x for x in cofactors])
    else:
        cofactors = None 
    rn_to_edges = defaultdict(list)
    with open(reaction_lst) as f:
        compound_re = re.compile("(C\d+)")
        for line in f:
            rn = line[:6]
            assert rn.startswith('R') and rn[1:].isdigit()
            lhs, rhs = line.split("<=>")
            reactants = compound_re.findall(lhs)
            products  = compound_re.findall(rhs)
            if cofactors is not None:
                reactants = [r for r in reactants if r not in cofactors]
                products = [p for p in products if p not in cofactors]
            for r in reactants:
                for p in products:
                    if r != p:
                        u, v = min(r, p), max(r, p)
                        rn_to_edges[rn].append((u,v))
    print('number of reactions', len(rn_to_edges))
    print('Average # edges reaction',
          np.mean([len(x) for x in rn_to_edges.values()]))
    rc_to_rns = defaultdict(list)
    with open(rclass_reaction_list) as f:
        for line in f:
            rc, rn = line.split()
            rc, rn = rc[3:], rn[3:]
            assert rc.startswith('RC') and len(rc) == 7, rc
            assert rn.startswith('R') and len(rn) == 6, rn
            rc_to_rns[rc].append(rn)
    print('Average # reactions per rclass',
          np.mean([len(x) for x in rc_to_rns.values()]))
    rc_to_edges = {}
    for rc, rns in rc_to_rns.items():
        rc_to_edges[rc] = []
        for rn in rns:
            for edge in rn_to_edges[rn]:
                rc_to_edges[rc].append(edge)
        if len(rc_to_edges[rc]) == 0:
            del rc_to_edges[rc]
    print('Average # edges per rclass', \
          np.mean([len(x) for x in rc_to_edges.values()]))
    with open(output_file, 'wb') as f:
        pickle.dump(rc_to_edges, f)
        print('Dumped to', output_file)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
            '--reaction_lst',
            default='%s/reaction/reaction.lst' % LIGAND)
    parser.add_argument(
            '--rclass_reaction_list',
            default='%srclass/links/rclass_reaction.list' % LIGAND)
    parser.add_argument(
            '--cofactor_list',
            default='%s/kegg/cofactor.lst' % os.environ['DATAPATH'])
    parser.add_argument(
            '--output_file',
            default='%s/kegg/rclass_to_edges.pkl' % os.environ['DATAPATH'])

    args = parser.parse_args() 
    compile_rclass(args.reaction_lst, args.rclass_reaction_list, 
                   args.output_file, args.cofactor_list)

