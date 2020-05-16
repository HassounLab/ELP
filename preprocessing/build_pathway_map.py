from argparse import ArgumentParser
import os
from pprint import pprint
import re
import pickle
import numpy as np


pathway_names = [
        'Carbohydrate metabolism',
        'Energy metabolism',
        'Lipid metabolism',
        'Nucleotide metabolism',
        'Amino acid metabolism',
        'Metabolism of other amino acids',
        'Glycan biosynthesis and metabolism',
        'Metabolism of cofactors and vitamins',
        'Metabolism of terpenoids and polyketides',
        'Biosynthesis of other secondary metabolites',
        'Xenobiotics biodegradation and metabolism']

def build_pathway_map(pathway_file, pathway_reaction_link_file, reaction_file, 
                      pathway_map_file, cofactors_file=None):
    # we assume that it is possible for one pathway to be in multiple pathway groups
    # or that one reaction can be in multiple pathways
    pnamenums = set([]) # set of pathway (name, numbers) we're tracking
    
    # this file lists pathway numbers associated with each pathway name
    with open(pathway_file) as f:
        flag = False
        for line in f:
            if line.startswith('##'):
                name = line[2:-1]
                if name in pathway_names:
                    flag = True
                else:
                    flag = False
            elif flag:
                pnum = line[:5]
                assert pnum.isdigit()
                pnamenums.add((name, pnum))
        for pname in pathway_names:
            if pname not in [pn for pn, _ in pnamenums]:
                raise RuntimeError('Error: %s not seen in %s' % (pname, pathway_file))
    print('Number of pathway (name, num) pairs tracked in', pathway_file, len(pnamenums))
    
    pprint(pnamenums)
    rn_to_path = {} # map reaction number to list of pathway (name, num)
    path_seen = set([])
    # this file links one pathway with one reaction on each line
    with open(pathway_reaction_link_file) as f:
        for line in f:
            if line.startswith('path:rn'):
                continue
            pnum, rn = line.split()
            pnum = pnum[8:]
            rn = rn[3:]
            assert len(pnum) == 5 and pnum.isdigit()
            assert len(rn) == 6 and rn.startswith('R') and rn[1:].isdigit()
            for pname, pn in pnamenums:
                if pn == pnum:
                    if rn in rn_to_path:
                        rn_to_path[rn].append((pname, pn))
                    else:
                        rn_to_path[rn] = [(pname, pn)]
                    path_seen.add(pname)

        for pname in pathway_names:
            if pname not in path_seen:
                print('Possible error: %s associated reactions not found in %s' \
                      % (pname, pathway_reaction_link_file))
    print('Total reactions tracking', len(rn_to_path))
    print('Mean number of pathways associated with each reaction', 
          np.mean([len(v) for v in rn_to_path.values()]))
    if cofactors_file is not None:
        print('Filtering for cofactors')
        with open(cofactors_file) as f:
            cofactors = f.read().splitlines()
        cofactors = set(['C' + '0' * (5 - len(x)) + x for x in cofactors])
    else:
        cofactors = None
    
    path_to_pairs = {} # map pathway (name, num) to list of compound pairs
    path_seen = set([])
    #pairs_to_pname = {} # map compound pair to list of pathway names
    # this is a file that lists reaction (numbers) and describes the associated reaction
    with open(reaction_file) as f:
        compound_re = re.compile("(C\d+)")
        for line in f:
            rn = line[:6]
            assert rn.startswith('R') and rn[1:].isdigit()
            if rn in rn_to_path:
                for path in rn_to_path[rn]:
                    if path not in path_to_pairs:
                        path_to_pairs[path] = []
                        path_seen.add(path[0])
                lhs, rhs = line.split("<=>")
                reactants = compound_re.findall(lhs)
                products  = compound_re.findall(rhs)
                if cofactors is not None:
                    reactants = [r for r in reactants if r not in cofactors]
                    products = [p for p in products if p not in cofactors]
                for r in reactants:
                    for p in products:
                        for path in rn_to_path[rn]:
                            path_to_pairs[path].append((r, p))
        for pname in pathway_names:
            if pname not in path_seen:
                print('Possible error: %s associated reactions not found in %s' \
                      % (pname, reaction_file))
    print('Number of compound pairs for each pathway name')
    for path, cpairs in path_to_pairs.items():
        print(path, len(cpairs))
    
    with open(pathway_map_file, 'wb') as f:
        pickle.dump(path_to_pairs, f)
        print('Dumped mappings to', pathway_map_file)
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
            '--pathway_list',
            default='%s/kegg/Pathway_April_20_2020/pathway.list' % os.environ['DATAPATH'])
    parser.add_argument(
            '--pathway_reaction_list',
            default='%s/kegg/Pathway_April_20_2020/links/pathway_reaction.list' \
                    % os.environ['DATAPATH'])
    parser.add_argument(
            '--reaction_list',
            default='%s/kegg/Ligand_April_20_2020/reaction/reaction.lst' \
                    % os.environ['DATAPATH'])
    parser.add_argument(
            '--cofactor_list',
            default='%s/kegg/cofactor.lst' % os.environ['DATAPATH'])
    parser.add_argument(
            '--pathway_map_file',
            default='%s/kegg/pathway_map.pkl' % os.environ['DATAPATH'])

    args = parser.parse_args()
    build_pathway_map(args.pathway_list, args.pathway_reaction_list, 
                      args.reaction_list, args.pathway_map_file, args.cofactor_list)
