import seaborn as sns
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr

csvspath = 'samplepath'
cmap = sns.cubehelix_palette(light=1, as_cmap=True)

filename = os.path.join(csvspath, 'csvsautocorrelations.csv')
try:
    dir = os.path.join(csvspath, 'boxplots')
    os.makedirs(dir)
except:
    pass

ticks = ['1', '2', '3', '4']

accs = []
taccs = []
loss = []
tloss = []

for i in range(10):
    df = pd.read_csv(os.path.join(csvspath, 'run_'+str(i)+'.csv'), sep=';')
    accs.append(np.median(df['acc'])), taccs.append(np.median(df['acc test']))
    loss.append(np.median(df['loss'])), tloss.append(np.median(df['loss test']))

acc_bounds = np.multiply(0.15, accs)
tacc_bounds = np.multiply(0.15, taccs)
loss_bounds = np.multiply(0.15, loss)
tloss_bounds = np.multiply(0.15, tloss)


sns.set_style('white')
sns.set_style('ticks', {'axes.edgecolor': '0',
                        'xtick.color': '0',
                        'ytick.color': '0'})

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


df = pd.read_csv(filename, sep=';')

splitA = df.loc[0:9]
splitAT = df.loc[10:19]
splitL = df.loc[20:29]
splitLT = df.loc[30:39]


splitA_ = [np.array(splitA.iloc[:,1]), np.array(splitA.iloc[:,2]),
           np.array(splitA.iloc[:,3]), np.array(splitA.iloc[:,4])]
splitAT_ = [np.array(splitAT.iloc[:,1]), np.array(splitAT.iloc[:,2]),
            np.array(splitAT.iloc[:,3]), np.array(splitAT.iloc[:,4])]
splitL_ = [np.array(splitL.iloc[:,1]), np.array(splitL.iloc[:,2]),
           np.array(splitL.iloc[:,3]), np.array(splitL.iloc[:,4])]
splitLT_ = [np.array(splitLT.iloc[:,1]), np.array(splitLT.iloc[:,2]),
            np.array(splitLT.iloc[:,3]), np.array(splitLT.iloc[:,4])]


####################################################################
#
# Standard deviation plots
#
####################################################################

"""for x in range(4):
    plt.errorbar(x+1, np.mean(splitA_[x]), np.std(splitA_[x]), linestyle='None', marker='.',barsabove=True,capsize=10)
plt.xlabel('steps')
plt.ylabel('autocorrelation')
plt.xticks(range(0,5))
plt.savefig(os.path.join(dir, 'mean_stdev_acc'+ ".png"))
plt.clf()

for x in range(4):
    plt.errorbar(x+1, np.mean(splitAT_[x]), np.std(splitAT_[x]), linestyle='None', marker='.',barsabove=True,capsize=10)
plt.xlabel('steps')
plt.ylabel('autocorrelation')
plt.xticks(range(0,5))
plt.savefig(os.path.join(dir, 'mean_stdev_test_acc'+ ".png"))
plt.clf()

for x in range(4):
    plt.errorbar(x+1, np.mean(splitL_[x]), np.std(splitL_[x]), linestyle='None', marker='.',barsabove=True,capsize=10)
plt.xlabel('steps')
plt.ylabel('autocorrelation')
plt.xticks(range(0,5))
plt.savefig(os.path.join(dir, 'mean_stdev_loss'+ ".png"))
plt.clf()

for x in range(4):
    plt.errorbar(x+1, np.mean(splitLT_[x]), np.std(splitLT_[x]), linestyle='None', marker='.',barsabove=True,capsize=10)
plt.xlabel('steps')
plt.ylabel('autocorrelation')
plt.xticks(range(0,5))
plt.savefig(os.path.join(dir, 'mean_stdev_test_loss'+ ".png"))
plt.clf()
"""


####################################################################
#
# Single boxplots
#
####################################################################

"""
box_acc = sns.boxplot(data=splitA_, color='white', width=.5)
box_acc.set(xlabel='steps', ylabel='autocorrelation')
plt.savefig(os.path.join(dir, 'acc'+ ".png"))
plt.clf()


box_acc = sns.boxplot(data=splitAT_, color='white', width=.5)
box_acc.set(xlabel='steps', ylabel='autocorrelation')
plt.savefig(os.path.join(dir, 'Tacc' + ".png"))
plt.clf()


box_acc = sns.boxplot(data=splitL_, color='white', width=.5)
box_acc.set(xlabel='steps', ylabel='autocorrelation')
plt.savefig(os.path.join(dir, 'loss' + ".png"))
plt.clf()


box_acc = sns.boxplot(data=splitLT_, color='white', width=.5)
box_acc.set(xlabel='steps', ylabel='autocorrelation')
plt.savefig(os.path.join(dir, 'Tloss' + ".png"))
plt.clf()
"""

####################################################################
#
# Joined boxplots
#
####################################################################


plt.figure(dpi=1000)
bpl = plt.boxplot(splitA_, positions=np.array(range(len(splitA_)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(splitAT_, positions=np.array(range(len(splitAT_)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, 'blue')
set_box_color(bpr, 'cadetblue')
plt.plot([], c='blue', label='Train')
plt.plot([], c='cadetblue', label='Test')
plt.axhline(y=0.15, linestyle="--", c='black', label='Threshold = 0.15',  linewidth=1)
plt.legend(fontsize='x-large')
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(-1, 1)
plt.ylabel('Autocorrelation value', fontsize=16)
plt.xlabel('Step size', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+'Acc' + ".eps"))
plt.clf()

plt.figure(dpi=1000)
bpl = plt.boxplot(splitL_, positions=np.array(range(len(splitL_)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(splitLT_, positions=np.array(range(len(splitLT_)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, 'blue')
set_box_color(bpr, 'cadetblue')
plt.plot([], c='blue', label='Train')
plt.plot([], c='cadetblue', label='Test')
plt.axhline(y=0.15, linestyle="--", c='black', label='Threshold = 0.15',  linewidth=1)
plt.legend()
plt.legend(fontsize='x-large')
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-2, len(ticks)*2)
plt.ylim(-1, 1)
plt.ylabel('Autocorrelation value', fontsize=16)
plt.xlabel('Step size', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(dir, dsetname+'_'+mutetype+'_'+'Loss' + ".eps"))
plt.clf()
