from argparse import ArgumentParser
import re
import os
import numpy as np
import pandas as pd
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
    df = pd.DataFrame(columns=['reactant', 'product', 'reaction_number'])

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
                        df.loc[len(df)] = [r, p, rn]
    rn_to_rc = {}
    with open(rclass_reaction_list) as f:
        for line in f:
            rc, rn = line.split()
            rc, rn = rc[3:], rn[3:]
            assert rc.startswith('RC') and len(rc) == 7, rc
            assert rn.startswith('R') and len(rn) == 6, rn
            rn_to_rc[rn] = rc
    df['rclass'] = df['reaction_number'].map(rn_to_rc)
    print(df['rclass'].value_counts(dropna=False))
    df.to_csv(output_file, index=False)
    print('Saved df to', output_file)
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
            default='%s/kegg/kegg_reactions.csv' % os.environ['DATAPATH'])

    args = parser.parse_args() 
    compile_rclass(args.reaction_lst, args.rclass_reaction_list, 
                   args.output_file, args.cofactor_list)

