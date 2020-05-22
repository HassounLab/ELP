from argparse import ArgumentParser
import re
import os
import networkx as nx

LIGAND = '%s/kegg/Ligand_April_20_2020/' % os.environ['DATAPATH']

def build_reaction_edgelist(reaction_file, output_edgelist=None, cofactor_file=None):
    G = nx.Graph()
          
    compound_re = re.compile("(C\d+)")
    with open(reaction_file) as f:
        for line in f:
            lhs, rhs = line.split("<=>")
            reactants = compound_re.findall(lhs)
            products  = compound_re.findall(rhs)
            for r in reactants:
                for p in products:
                    if r != p:
                        G.add_edge(r, p)
    print('All reactions')
    print(nx.info(G)) 
    if cofactor_file is not None:
        with open(cofactor_file, encoding='utf-8-sig') as f:
            cofactors = f.read().splitlines()
        cofactors = ['C' + '0' * (5 - len(x)) + x for x in cofactors]
        cofactors = [c for c in cofactors if c in G.nodes]
        print('Removing %d cofactors in G' % len(cofactors))
        G.remove_nodes_from(cofactors)
        print('After filtering cofactors')
        print(nx.info(G))
    if output_edgelist is not None:
        nx.write_edgelist(G, output_edgelist)
        print('Saved to', output_edgelist)
    return G


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
            '--reaction_lst',
            default='%s/reaction/reaction.lst' % LIGAND)
    parser.add_argument(
            '--kegg_edgelist',
            default= '%s/kegg/kegg_2020.edgelist' % os.environ['DATAPATH'])
    parser.add_argument(
            '--cofactor_lst',
            default='%s/kegg/cofactor.lst' % os.environ['DATAPATH'])
    args = parser.parse_args()
    build_reaction_edgelist(args.reaction_lst, args.kegg_edgelist, args.cofactor_lst)
    
