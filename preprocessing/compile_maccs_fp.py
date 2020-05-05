from argparse import ArgumentParser
import re
import os
import networkx as nx
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import MACCSkeys 
from rdkit import DataStructs

def get_mol(mol_dir, compound_number):
    molfile = os.path.join(mol_dir, compound_number + '.mol')
    if not os.path.exists(molfile):
        print("Cannot find *.mol file for %s." % compound_number)
        return None
    m = Chem.MolFromMolFile(molfile)
    if m is None:
        print("could not generate Mol from *.mol file for node", compound_number)
    return m

def get_maccs_fingerprint_from_mol(m):
    if m is None: return None
    k = MACCSkeys.GenMACCSKeys(m)
    textk = DataStructs.BitVectToText(k)
    binaryvec = np.array(list(map(int, textk)))
    return binaryvec

def get_pubchem_fingerprint_from_mol(m):
    if m is None: return None
    smile = Chem.MolToSmiles(m)
    return smile

def compile_maccs_fingerprints(compounds, mol_dir, output_file=None):
    maccs = {}
    for c in compounds:
        fp = get_maccs_fingerprint_from_mol(get_mol(mol_dir, c))
        if fp is not None:
            maccs[c] = fp
    print('%d/%d compounds have MACCS fingerprints' % (len(maccs), len(compounds)))
    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump(maccs, f)
            print('Dumped to', output_file)
    return maccs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
           '--mol_dir', 
            help='A valid path to the directory ligand/compound/mol, ' + \
                 'where each compound has a CXXXXX.mol file.',
            default='%s/kegg/Ligand_April_20_2020/compound/mol'\
                     % os.environ['DATAPATH'])
    parser.add_argument(
            '--kegg_edgelist',
            default='%s/kegg/kegg_2020.edgelist' % os.environ['DATAPATH'])
    parser.add_argument(
            '--maccs_file',
            default='%s/kegg/kegg_2020_maccs_fp.pkl' % os.environ['DATAPATH'])
    args = parser.parse_args() 
    G = nx.read_edgelist(args.kegg_edgelist)
    compile_maccs_fingerprints(G.nodes, args.mol_dir, args.maccs_file)
