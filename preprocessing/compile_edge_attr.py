from argparse import ArgumentParser
from collections import defaultdict
from operator import itemgetter
import re
import os
import pickle
import numpy as np
import pandas as pd
import random
LIGAND = '%s/kegg/Ligand_April_20_2020/' % os.environ['DATAPATH']

def compile_edge_attr(reaction_lst, el_reaction_list, output_file,
                      edge_type, cofactors_file=None):
    assert edge_type in ['rc', 'ec']
    print(cofactors_file)
    if cofactors_file is not None:
        print('Filtering for cofactors')
        with open(cofactors_file, encoding='utf-8-sig') as f:
            cofactors = f.read().splitlines()
        cofactors = set(['C' + '0' * (5 - len(x)) + x for x in cofactors])
    else:
        cofactors = None
    edge_to_rns = defaultdict(list)
 
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
                        edge_to_rns[(r, p)].append(rn)
    print('number of edges', len(edge_to_rns))
    print('Average # reaction numbers per edge',
          np.mean([len(x) for x in edge_to_rns.values()]))
    rn_to_els = defaultdict(list)
    with open(el_reaction_list) as f:
        for line in f:
            el, rn = line.split()
            el, rn = el[3:], rn[3:]
            if edge_type == 'ec':
                el = '.'.join(el.split('.')[:-1])
                assert sum(1 for x in el if x == '.') == 2, el
            elif edge_type == 'rc':
                assert el.startswith('RC') and len(el) == 7, el
            assert rn.startswith('R') and len(rn) == 6, rn
            rn_to_els[rn].append(el)        
    print('Average # edge labels numbers per reaction number',
          np.mean([len(x) for x in rn_to_els.values()]))
    """"
    edge_to_els = {}
    for edge, rns in edge_to_rns.items():
        edge_to_els[edge] = []
        for rn in rns:
            if rn not in rn_to_els:
                continue
            for el in rn_to_els[rn]:
                if el not in edge_to_els[edge]:
                    edge_to_els[edge].append(el)
        if len(edge_to_els[edge]) == 0:
            del edge_to_els[edge]
    print('# edges with edge labels', len(edge_to_els))
    el_counts = [len(x) for x in edge_to_els.values()]
    print('Edge label count distribution: mean %f median %f max %f'\
          % (np.mean(el_counts), np.median(el_counts), np.max(el_counts)))
    """
    edge_to_el = {}
    num_ambiguous = 0
    for (u, v), rns in edge_to_rns.items():
        el_counts = defaultdict(int)
        for rn in rns:
            if rn not in rn_to_els:
                continue
            for el in rn_to_els[rn]:
                el_counts[el] += 1
        if len(el_counts) == 0:
            continue
        max_el_count = max(el_counts.values())
        els = [el for el, c in el_counts.items() if c == max_el_count]
        if len(els) > 1:
            num_ambiguous += 1
            max_el = random.choice(els)
        else:
            max_el = els[0]
        edge_to_el[(u,v)] = max_el
    print('Num ambiguous edge labels', num_ambiguous)
    print('# edges with edge label', len(edge_to_el))
    with open(output_file, 'wb') as f:
        pickle.dump(edge_to_el, f)
        print('Dumped to', output_file)
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
            '--reaction_lst',
            default='%s/reaction/reaction.lst' % LIGAND)
    parser.add_argument('--edge_type', default='rc')
    parser.add_argument(
            '--rclass_reaction_list',
            default='%srclass/links/rclass_reaction.list' % LIGAND)
    parser.add_argument(
            '--eclass_reaction_list',
            default='%senzyme/links/enzyme_reaction.list' % LIGAND)
    parser.add_argument(
            '--cofactor_list',
            default='%s/kegg/cofactor.lst' % os.environ['DATAPATH'])

    args = parser.parse_args() 
    print(args.edge_type)
    assert args.edge_type in ['rc', 'ec']
    if args.edge_type == 'rc':
        el_reaction_list = args.rclass_reaction_list
    else:
        el_reaction_list = args.eclass_reaction_list
    edge_attr_file = '%s/kegg/kegg_2020_%s_edge.pkl' \
                     % (os.environ['DATAPATH'], args.edge_type)
    compile_edge_attr(args.reaction_lst, el_reaction_list, 
                      edge_attr_file, args.edge_type, args.cofactor_list)
