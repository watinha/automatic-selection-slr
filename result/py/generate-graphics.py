import seaborn, sys, np, scipy

import pandas as pd
import matplotlib.pyplot as plt

if (len(sys.argv) < 2):
    print('You should inform a CSV file...')
    sys.exit(1)

if (len(sys.argv) < 3):
    print('You should inform a number of features (k50, k100, k300, k1000, k3000)...')
    sys.exit(1)

if (len(sys.argv) < 4):
    print('You should inform the metric which is been evaluated...')
    sys.exit(1)

csv = sys.argv[1]
k = sys.argv[2]
metric = sys.argv[3]
data = pd.read_csv(csv)

seaborn.set(rc={'figure.figsize':(4, 5)})
seaborn.distplot(data[('SVM_%s' % k)], bins=40, hist=False,
        kde=True, kde_kws={'shade': True,  'linewidth': 3}, label='SVM')
seaborn.distplot(data[('DT_%s' % k)], bins=40, hist=False,
        kde=True, kde_kws={'shade': True,  'linewidth': 3}, label='DT')

plt.ylabel('Frequency')
plt.xlabel(('%s with %s' % (metric, k)))
plt.ylim(0, 5)
plt.xlim(0, 1)
plt.savefig('%s-%s.png' % (metric, k))

sys.exit(0)
