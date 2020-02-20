from scipy import stats
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.ticker as ticker

dsetname = 'CIFAR10'
experiment = 'Sample'
mutetype = 'topology'

maindir = os.path.join('results', dsetname, experiment, mutetype)

accs = []
taccs = []
loss = []
tloss = []

df = pd.read_csv(os.path.join(maindir, 'sample.csv'), sep=',')
accs.append(np.array(df['acc'])), taccs.append(np.array(df['acc test']))
loss.append(np.array(df['loss'])), tloss.append(np.array(df['loss test']))

loss = np.round(loss, decimals=2).tolist()[0]
print('loss: ', loss)

plt.hist(loss, alpha=0.5,density=False, ec='black', histtype='bar')


plt.ylabel('Frequency', fontsize=16)
plt.xlabel('Loss Values', fontsize=16)
plt.tight_layout()
#plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+'Loss_New' + ".eps"))
#plt.clf()
plt.show()
plt.clf()
sns.distplot(loss, kde=True, bins=10, hist_kws=dict(edgecolor="k", linewidth=1))
#sns.jointplot(x="x", y="y", data=loss, kind="kde");
plt.show()