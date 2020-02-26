from scipy import stats
import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.ticker as ticker

csvspath = 'samplepath'

accs = []
taccs = []
loss = []
tloss = []

df = pd.read_csv(os.path.join(csvpath, 'sample.csv'), sep=',')
accs.append(np.array(df['acc'])), taccs.append(np.array(df['acc test']))
loss.append(np.array(df['loss'])), tloss.append(np.array(df['loss test']))
df = pd.read_csv(os.path.join(csvpath, 'sample_pairs.csv'), sep=',')
accs.append(np.array(df['acc'])), taccs.append(np.array(df['acc test']))
loss.append(np.array(df['loss'])), tloss.append(np.array(df['loss test']))


try:
    dir = os.path.join(csvpath, 'density')
    os.makedirs(dir)
except:
    pass

x = loss[0]
y = loss[1]

plt.figure(dpi=1000)
sns.kdeplot(x,y, shade=True, kernel='gau')
plt.xlabel('Fitness', fontsize=16)
plt.ylabel('Fitness Neighbors', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+ 'all_loss' + ".eps"))
plt.clf()
"""
x = tloss[0]
y = tloss[1]

plt.figure(dpi=1000)
sns.kdeplot(x,y, shade=True, cmap="Oranges", kernel='gau')
plt.xlabel('Fitness', fontsize=16)
plt.ylabel('Fitness Neighbors', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+ 'all_test_loss' + ".eps"))
plt.clf()
"""