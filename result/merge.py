import sys

import pandas as pd

files = ['games.csv', 'mdwe.csv', 'pair.csv', 'testing.csv',
         'illiterate.csv', 'ontologies.csv', 'slr.csv', 'xbi.csv']

argument = sys.argv[1]
if argument == 'precision':
    columns = [0, 1, 2] # precision
if argument == 'recall':
    columns = [0, 3, 4] # recall
if argument == 'fscore':
    columns = [0, 5, 6] # fscore

#columns = [0, 9, 10] # threashold
#columns = [0, 13, 14] # fscore with threashold

if argument == 'exclusion':
    columns = [0, 7, 8] # correct exclusion
if argument == 'missed':
    columns = [0, 11, 12] # missed
if argument == 'exclusion-baseline':
    columns = [0, 15, 16] # exclusion baseline
if argument == 'missed-baseline':
    columns = [0, 17, 18] # missed baseline

folders = [ 'k50', 'k100', 'k300', 'k1000', 'k3000' ]

complete_table = pd.DataFrame()
for folder in folders:
    table = pd.DataFrame()
    for f in files:
        frame = pd.read_csv('%s/%s' % (folder, f), sep=';')
        print('%s/%s' % (folder, f))
        if table.size == 0:
            table = frame.take(columns, axis=1)
            table['group'] = f
        else:
            aux = frame.take(columns, axis=1)
            aux['group'] = f
            table = table.append(aux)

    if complete_table.size == 0:
        complete_table = table
    else:
        complete_table = complete_table.merge(table, left_on=['fold', 'group'], right_on=['fold', 'group'])

f = open('all-%s.csv' % (argument), 'w')
f.write(complete_table.to_csv())
f.close()
