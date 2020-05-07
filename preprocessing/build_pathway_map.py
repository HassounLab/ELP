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
    pnums = set([]) # set of pathway numbers we're tracking
    pname_to_pnum = {} # map pathway names to pathway numbers
    # we assume that it is possible for one pathway to be in multiple pathway groups
    # or that one reaction can be in multiple pathways

    with open(pathway_file) as f:
        flag = False
        for line in f:
            if line.startswith('##'):
                name = line[2:-1]
                if name in pathway_names:
                    flag = True
                    pname_to_pnum[name] = []
                else:
                    flag = False
            elif flag:
                pnum = line[:5]
                assert pnum.isdigit()
                pnums.add(pnum)
                pname_to_pnum[name].append(pnum)
        for pname in pathway_names:
            if pname not in pname_to_pnum:
                raise RuntimeError('Error: %s not seen in %s' % (pname, pathway_file))

    print('Number of pathway numbers tracking', len(pnums))

    rn_to_pname= {} # map reaction number to list of pathway names
    pname_to_rn = {} # map pathway names to reaction numbers
    with open(pathway_reaction_link_file) as f:
        for line in f:
            if line.startswith('path:rn'):
                continue
            pnum, rn = line.split()
            pnum = pnum[8:]
            rn = rn[3:]
            assert len(pnum) == 5 and pnum.isdigit()
            assert len(rn) == 6 and rn.startswith('R') and rn[1:].isdigit()
            if pnum in pnums:
                for pname, pns in pname_to_pnum.items():
                    if pnum in pns:
                        if pname in pname_to_rn:
                            pname_to_rn[pname].append(rn)
                        else:
                            pname_to_rn[pname] = [rn]

                        if rn in rn_to_pname:
                            rn_to_pname[rn].append(pname)
                        else:
                            rn_to_pname[rn] = [pname]

        for pname in pathway_names:
            if pname not in pname_to_rn:
                print('Possible error: %s associated reactions not found in %s' \
                      % (pname, pathway_reaction_link_file))
    print('Total reactions tracking', len(rn_to_pname))

    print('Number of reactions for each pathway name')
    for pname, rns in pname_to_rn.items():
        print(pname, len(rns))
    
    if cofactors_file is not None:
        print('Filtering for cofactors')
        with open(cofactors_file) as f:
            cofactors = f.read().splitlines()
        cofactors = set(['C' + '0' * (5 - len(x)) + x for x in cofactors])
    else:
        cofactors = None
    
    pname_to_pairs = {} # map pathway name to list of compound pairs
    pairs_to_pname = {} # map compound pair to list of pathway names
    with open(reaction_file) as f:
        compound_re = re.compile("(C\d+)")
        for line in f:
            rn = line[:6]
            assert rn.startswith('R') and rn[1:].isdigit()
            if rn in rn_to_pname:
                pnames = rn_to_pname[rn]
                for pname in pnames:
                    if pname not in pname_to_pairs:
                        pname_to_pairs[pname] = []
                lhs, rhs = line.split("<=>")
                reactants = compound_re.findall(lhs)
                products  = compound_re.findall(rhs)
                if cofactors is not None:
                    reactants = [r for r in reactants if r not in cofactors]
                    products = [p for p in products if p not in cofactors]
                for r in reactants:
                    for p in products:
                        pname_to_pairs[pname].append((r, p))
                        if (r, p) not in pairs_to_pname:
                            pairs_to_pname[(r, p)] = [pname]
                        else:
                            pairs_to_pname[(r, p)].append(pname)
        for pname in pathway_names:
            if pname not in pname_to_pairs:
                print('Possible error: %s associated reaction pairs not found in %s'\
                      % (pname, reaction_file))
    print('Number of reaction pairs tracking:', len(pairs_to_pname))
    print('Number of compound pairs for each pathway name')
    for pname, cpairs in pname_to_pairs.items():
        print(pname, len(cpairs))
    
    print('Average associated pathways', np.mean([len(x) for x in pairs_to_pname]))
    
    with open(pathway_map_file, 'wb') as f:
        obj = {'pname_to_pairs': pname_to_pairs, 'pairs_to_pname': pairs_to_pname}
        pickle.dump(obj, f)
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
