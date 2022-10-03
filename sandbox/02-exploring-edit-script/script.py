'''
produced artifacts are saved to current dir
'''

from pathlib import Path
import pickle
import os
import json



def extract_data():
    '''
    extract records from data0.pkl
    '''
    
    # versions for which ground truth edit script can be generated
    versions = [7, 12, 32, 43, 50, 55, 61, 69]
    data0_path = Path('/code2/repo-recoder/sandbox/01-exploring-pkl-files/data0.pkl')

    data = pickle.load(open(data0_path, 'rb'))

    for ver in versions:
        buggy = data[ver]['old']
        fixed = data[ver]['new']
        buggytree = data[ver]['oldtree']
        fixedtree = data[ver]['newtree']

        os.makedirs(str(ver))
        with open(f'{ver}/{ver}-buggy.java', 'w') as fout:
            fout.write(buggy.replace(r'\n', '\n'))
        with open(f'{ver}/{ver}-fixed.java', 'w') as fout:
            fout.write(fixed.replace(r'\n', '\n'))
        with open(f'{ver}/{ver}-buggy.ast.txt', 'w') as fout:
            fout.write(buggytree.replace(' ', '\n'))
        with open(f'{ver}/{ver}-fixed.ast.txt', 'w') as fout:
            fout.write(fixedtree.replace(' ', '\n'))


def extract_edit_script():
    '''
    save files as json
    files that are generated from runsolvereplace.py
    '''

    process_datacopy_dir = Path('/code2/repo-recoder/newdata/')
    
    data = pickle.load(open(process_datacopy_dir / 'process_datacopy0.pkl', 'rb'))
    with open('process_datacopy.json', 'w') as fout:
        fout.write(json.dumps(data))

    data = pickle.load(open(process_datacopy_dir / 'rule0.pkl', 'rb'))
    with open('rule.json', 'w') as fout:
        fout.write(json.dumps(data))


extract_edit_script()
# extract_data()
