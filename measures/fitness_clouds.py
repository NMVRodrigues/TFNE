from scipy import stats
import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.ticker as ticker

dsetname = 'CIFAR10'
experiment = 'Sample'
mutetype = 'learning'

maindir = os.path.join('results', dsetname, experiment, mutetype)

accs = []
taccs = []
loss = []
tloss = []

df = pd.read_csv(os.path.join(maindir, 'sample.csv'), sep=',')
accs.append(np.array(df['acc'])), taccs.append(np.array(df['acc test']))
loss.append(np.array(df['loss'])), tloss.append(np.array(df['loss test']))
df = pd.read_csv(os.path.join(maindir, 'sample_pairs.csv'), sep=',')
accs.append(np.array(df['acc'])), taccs.append(np.array(df['acc test']))
loss.append(np.array(df['loss'])), tloss.append(np.array(df['loss test']))


try:
    dir = os.path.join(maindir, 'clouds')
    os.makedirs(dir)
except:
    pass

#or run in range(10):
#    x = accs[run][:-1]
#    y = accs[run][1:]

#    plt.scatter(x, y)
#    identity_line = np.linspace(max(min(x), min(y)),
#                                min(max(x), max(y)))
#    plt.plot(identity_line, identity_line, color="black", linestyle="-", linewidth=1.0)

#    plt.savefig(os.path.join(dir1, 'run'+str(run) + '_' + ".png"))
#    plt.clf()

#    x = taccs[run][:-1]
#    y = taccs[run][1:]

#    plt.scatter(x, y)
#    identity_line = np.linspace(max(min(x), min(y)),
#                                min(max(x), max(y)))
#    plt.plot(identity_line, identity_line, color="black", linestyle="-", linewidth=1.0)

#    plt.savefig(os.path.join(dir2, 'run'+str(run) + '_'  ".png"))
#    plt.clf()

x = accs[0]
y = accs[1]

plt.figure(dpi=1000)
identity_line = np.linspace(0,1)
plt.plot(identity_line, identity_line, color="black", linestyle="-", linewidth=1.0)
sns.kdeplot(x,y, shade=True, kernel='gau', bw='silverman')
plt.xlabel('Fitness', fontsize=18)
plt.ylabel('Fitness Neighbors', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+ 'all_acc' + ".eps"))
plt.clf()

#plt.figure(dpi=1000)
#plt.scatter(x, y, color='blue', s=8)
#identity_line = np.linspace(0,1)
#plt.plot(identity_line, identity_line, color="black", linestyle="-", linewidth=1.0)
#plt.xlabel('Fitness', fontsize=16)
#plt.ylabel('Fitness Neighbors', fontsize=16)
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+ 'all_acc' + ".eps"))
#plt.clf()


x = taccs[0]
y = taccs[1]
plt.figure(dpi=1000)
plt.scatter(x, y, color='orange', s=8)
identity_line = np.linspace(0,1)
plt.plot(identity_line, identity_line, color="black", linestyle="-", linewidth=1.0)
plt.xlabel('Fitness', fontsize=16)
plt.ylabel('Fitness Neighbors', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+ 'all_test_acc' + ".eps"))
plt.clf()

x = accs[0]
y = accs[1]
xt = taccs[0]
yt = taccs[1]

plt.figure(dpi=1000)
plt.scatter(x, y, color='blue', s=8)
plt.scatter(xt, yt, color='orange', s=8)
plt.legend(['Training', 'Test'])
identity_line = np.linspace(0,1)
plt.plot(identity_line, identity_line, color="black", linestyle="-", linewidth=1.0)
plt.xlabel('Fitness', fontsize=16)
plt.ylabel('Fitness Neighbors', fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+ 'all_both_acc' + ".eps"))
plt.clf()

#    x = loss[run][:-1]
#    y = loss[run][1:]

#    plt.scatter(x, y)
#    identity_line = np.linspace(max(min(x), min(y)),
#                                min(max(x), max(y)))
#    plt.plot(identity_line, identity_line, color="black", linestyle="-", linewidth=1.0)

#    plt.savefig(os.path.join(dir2, 'run'+str(run) + '_' + 'loss' ".png"))
#    plt.clf()


