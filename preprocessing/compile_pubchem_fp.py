from argparse import ArgumentParser
from time import sleep
import requests
import base64
import sys
import networkx as nx
import numpy as np
import pickle
import os


def convert_to_binary_vec(fpbytes):
    s = ''.join(["{:08b}".format(x) for x in fpbytes]) # binary string
    return np.array(list(map(int, s)))


def get_pubchem_fingerprints_from_compound(compound):
    assert(len(compound) == 6 and compound[0] == 'C' and compound[1:5].isdigit())

    sleep(0.5)
    try:
        response = requests.get(
            'https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/'\
                    'sourceid/KEGG/%s/cids/JSON' % compound,
            timeout=30)
    except requests.exceptions.Timeout:
        print('CID request timed out for', compound, file=sys.stderr)
        return None

    if response.status_code != 200:
        print('CID request status code returned', response.status_code,
              'for', compound, file=sys.stderr)
        return None

    cids = [int(cid) for x in response.json()['InformationList']['Information'] \
                    for cid in x['CID']]

    if len(cids) == 0:
        print('No matches found for', compound, file=sys.stderr)
        return None
    elif len(cids) > 100:
        print('Too many matches found', compound, file=sys.stderr)
        return None

    sleep(0.5)
    try:
        response = requests.post(
                'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'\
                        'property/Fingerprint2D/JSON',
                 data={'cid': ','.join('%d' % cid for cid in cids)},
                 timeout=30)
    except requests.exceptions.Timeout:
        print('Fingerpring request timed out for', compound, file=sys.stderr)
        return None
    if response.status_code != 200:
        print('Fingerprint requet status code returned', response.status_code, 
              'for', compound, file=sys.stderr)
        return None
    fingerprints = [] # cid: fingerprint bytes
    for properties in response.json()['PropertyTable']['Properties']:
        fingerprints.append(
                (properties['CID'], base64.b64decode(properties['Fingerprint2D'])))
  

    if len(fingerprints) > 1:
        print('Ambiguous pubchem fingerprints for', cid)
        for cid in sorted(fingerprints.keys()):
            print('    * cid %d: %s' % (cid, fingerprints[cid].hex()))
        print()
    return convert_to_binary_vec(fingerprints[0][1])
def compile_pubchem_fingerprints(compounds, output_file):
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            pubchem = pickle.load(f)
        print('Using previously stored pubchem fingerprints')
        print('%d fingerprints are already available' % (len(pubchem)))
    else:
        pubchem = {}
    num_processed = len(pubchem)
    for n in compounds:
        if n not in pubchem:
            fp = get_pubchem_fingerprints_from_compound(n)
            if fp is not None:
                pubchem[n] = fp
    print('got pubchem fingerprints for %d compounds' % (len(pubchem) - num_processed))
    print('%d/%d compounds have Pubchem fingerprints' % (len(pubchem), len(compounds)))
    with open(output_file, 'wb') as f:
        pickle.dump(pubchem, f)
        print('Dumped to', output_file)
    return pubchem


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
            '--kegg_edgelist',
            default='%s/kegg/kegg_2020.edgelist' % os.environ['DATAPATH'])
    parser.add_argument(
            '--pubchem_file',
            default='%s/kegg/kegg_2020_pubchem_fp.pkl' % os.environ['DATAPATH'])
    args = parser.parse_args()
    G = nx.read_edgelist(args.kegg_edgelist)
    compile_pubchem_fingerprints(G.nodes, args.pubchem_file)

    
    



