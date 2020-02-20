from scipy import stats, mean, median
import seaborn as sns
import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as ticker

dsetname = 'SVHN'
walktype = 'Evolution'
evotype = 'loss'
mutetype = 'topology'

maindir = os.path.join('results', dsetname, walktype, mutetype, evotype, 'csvs')

accs = []
taccs = []
loss = []
tloss = []

for i in range(10):
    df = pd.read_csv(os.path.join(maindir, 'run_'+str(i)+'.csv'), sep=';')
    accs.append(df['acc']), taccs.append(df['acc test'])
    loss.append(df['loss']), tloss.append(df['loss test'])

try:
    dir = os.path.join('results', dsetname, walktype, mutetype, evotype, 'plots')
    os.makedirs(dir)
except:
    pass


'''plt.figure(dpi=1000)
for x in range(10):
    plt.plot(accs[x], linestyle="--")
plt.xticks(range(0,22, 2))
plt.xlabel('Generations', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+'Acc' + ".png"))
plt.clf()

plt.figure(dpi=1000)
for x in range(10):
    plt.plot(taccs[x], linestyle="--")
plt.xticks(range(0,22, 2))
plt.xlabel('Generations', fontsize=16)
plt.ylabel('Test Accuracy', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+'test_Acc' + ".png"))
plt.clf()'''

fig, (ax1, ax2) = plt.subplots(1, 2, sharey='col',
                        gridspec_kw={'wspace': 0.1})
for x in range(10):
    ax1.plot(loss[x], linestyle="--")
    ax2.plot(tloss[x], linestyle="-")
ax1.set_xticks([0,20])
ax2.set_xticks([0,20])
ax2.set_yticks([]) 
ax1.set_xlabel('Generations', fontsize=16)
ax1.set_ylabel('Loss', fontsize=16)
ax2.set_xlabel('Generations', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
#fig.xticks([0,20])
#fig.xlabel('Generations', fontsize=16)
#fig.ylabel('Loss', fontsize=16)
#fig.xticks(fontsize=15)
#fig.yticks(fontsize=15)
fig.dpi = 1000
fig.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+'Loss' + ".eps"))
#fig.clf()

'''plt.figure(dpi=1000)
for x in range(10):
    plt.plot(tloss[x], linestyle="--")
plt.xticks(range(0,22, 2))
plt.xlabel('Generations', fontsize=16)
plt.ylabel('Test Loss', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+'test_Loss' + ".png"))
plt.clf()'''


''' for x in range(10):
    plt.plot(accs[x], c='blue')
    plt.plot(taccs[x], c='orange')
    plt.legend(['Training', 'Test'])
    plt.xlabel('Generations')
    plt.ylabel('Accuracy')

    plt.savefig(os.path.join(dir, 'run_A' +str(x) + ".png"))
    plt.clf()

    plt.plot(np.cumsum(np.diff(accs[x])), c='blue')
    plt.plot(np.cumsum(np.diff(taccs[x])), c='orange')
    plt.legend(['Training', 'Test'])
    plt.xlabel('Generations')
    plt.ylabel('Accuracy difference ')

    plt.savefig(os.path.join(dir, 'dif_A' + str(x) + ".png"))
    plt.clf()

    plt.plot(loss[x], c='blue')
    plt.plot(tloss[x], c='orange')
    plt.legend(['Training', 'Test'])
    plt.xlabel('Generations')
    plt.ylabel('Loss')

    plt.savefig(os.path.join(dir, 'run_L' + str(x) + ".png"))
    plt.clf()

    plt.plot(np.cumsum(np.diff(loss[x])), c='blue')
    plt.plot(np.cumsum(np.diff(tloss[x])), c='orange')
    plt.legend(['Training', 'Test'])
    plt.xlabel('Generations')
    plt.ylabel('Loss')

    plt.savefig(os.path.join(dir, 'dif_L' + str(x) + ".png"))
    plt.clf() '''




