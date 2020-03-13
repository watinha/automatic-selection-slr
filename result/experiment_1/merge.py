import pandas as pd

files = ['games.csv', 'mdwe.csv', 'pair.csv', 'testing.csv',
         'illiterate.csv', 'ontologies.csv', 'slr.csv', 'xbi.csv']
#columns = [0, 1, 2] # precision
#columns = [0, 3, 4] # recall
columns = [0, 5, 6] # fscore
#columns = [0, 7, 8] # correct exclusion
#columns = [0, 9, 10] # threashold
#columns = [0, 11, 12] # missed
#columns = [0, 13, 14] # fscore with threashold
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

f = open('all-fscore.csv', 'w')
f.write(complete_table.to_csv())
f.close()
