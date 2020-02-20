from scipy import stats, mean, median
import seaborn as sns
import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as ticker

dsetname = 'FMNIST'
walktype = 'SelectiveWalk'
evotype = 'loss'
mutetype = 'learning'

maindir = os.path.join('results', dsetname, walktype, mutetype, evotype, 'csvs')

accs = []
taccs = []
Loss = []
tLoss = []

for i in range(10):
    df = pd.read_csv(os.path.join(maindir, 'run_'+str(i)+'.csv'), sep=';')
    accs.append(df['acc']), taccs.append(df['acc test'])
    Loss.append(df['loss']), tLoss.append(df['loss test'])

try:
    dir = os.path.join('results', dsetname, walktype, mutetype, evotype, 'overfit')
    os.makedirs(dir)
except:
    pass

plt.figure(dpi=1000)

for index, loss in enumerate(Loss):
    overfit = []
    btp = tLoss[index][0]
    tbtp = loss[0]

    for i in range(len(loss)):
        if i == 0:
            pass
        else:
            if loss[i] > tLoss[index][i]:
                overfit.append(0)
            else:
                if tLoss[index][i] < btp:
                    overfit.append(0)
                    btp = tLoss[index][i]
                    tbtp = loss[i]
                else:
                    #print('loss: ', loss[i])
                    #print('test loss: ', tLoss[index][i])
                    #print('btp: ', btp)
                    #print('tbtp: ', tbtp)
                    temp = abs(loss[i]-tLoss[index][i]) - abs(tbtp - btp)
                    if temp>=0:
                        overfit.append(temp)
                    else:
                        overfit.append(0)
                    #print('overfit: ', overfit[-1], '\n')

    plt.plot(overfit, linestyle="--", linewidth=4.0)
#plt.xticks(range(0,22, 2))
plt.xlabel('Walk Steps', fontsize=16)
plt.ylabel('Overfitting', fontsize=16)
plt.xlim(0, 30)
plt.ylim(0, 0.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+'overfit' + ".eps"))
plt.clf()
