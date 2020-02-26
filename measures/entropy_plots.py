import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import os


dsets = ['MNIST', 'FMNIST','CIFAR10', 'SVHN']
#dsetname = 'SVHN'
#walktype = 'SelectiveWalk'

#maindir = os.path.join('results', dsetname, walktype)
learning, params, topology = [],[],[]

for dset in dsets:
    maindir = os.path.join('results', dset, 'SelectiveWalk')
    learning.append(pd.read_csv(os.path.join(maindir, 'learning', 'loss', 'csvsEntropy.csv'), sep=';').iloc[11])
    params.append(pd.read_csv(os.path.join(maindir, 'params', 'loss', 'csvsEntropy.csv'), sep=';').iloc[11])
    topology.append(pd.read_csv(os.path.join(maindir, 'topology', 'loss', 'csvsEntropy.csv'), sep=';').iloc[11])

#learning = pd.read_csv(os.path.join(maindir, 'learning', 'loss', 'csvsEntropy.csv'), sep=';')
#params = pd.read_csv(os.path.join(maindir, 'params', 'loss', 'csvsEntropy.csv'), sep=';')
#topology = pd.read_csv(os.path.join(maindir, 'topology', 'loss', 'csvsEntropy.csv'), sep=';')

#ticks = ['0', '$\u03B5^*/128$', '$\u03B5^*/64$', '$\u03B5^*/32$', '$\u03B5^*/16$', '$\u03B5^*/8$', '$\u03B5^*/4$', '$\u03B5^*/2$', '$\u03B5^*$']
ticks = ['0', r'$\frac{ε^*}{128}$', r'$\frac{ε^*}{64}$', r'$\frac{ε^*}{32}$', r'$\frac{ε^*}{16}$', r'$\frac{ε^*}{8}$', r'$\frac{ε^*}{4}$', r'$\frac{ε^*}{2}$', r'$ε^*$']

plt.figure(dpi=1000)
plt.ylabel('H(ε)', fontsize=20)
plt.xlabel('ε', fontsize=20)
plt.xticks([1,2,3,4,5,6,7,8,9],ticks)
plt.ylim(0, 1)
plt.plot([1,2,3,4,5,6,7,8,9], learning[0], marker='.', linestyle=':', linewidth=1,  label='MNIST')
plt.plot([1,2,3,4,5,6,7,8,9], learning[1], marker='1', linestyle=':', linewidth=1,  label='FMNIST')
plt.plot([1,2,3,4,5,6,7,8,9], learning[2], marker='+', linestyle=':', linewidth=1,  label='CIFAR10')
plt.plot([1,2,3,4,5,6,7,8,9], learning[3], marker='x', linestyle=':', linewidth=1,  label='SVHN')
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.xticks(fontsize=20)
plt.savefig('learning.png')

plt.figure(dpi=1000)
plt.ylabel('H(ε)', fontsize=20)
plt.xlabel('ε', fontsize=20)
plt.xticks([1,2,3,4,5,6,7,8,9],ticks)
plt.ylim(0, 1)
plt.plot([1,2,3,4,5,6,7,8,9], params[0], marker='.', linestyle=':', linewidth=1, label='MNIST')
plt.plot([1,2,3,4,5,6,7,8,9], params[1], marker='1', linestyle=':', linewidth=1, label='FMNIST')
plt.plot([1,2,3,4,5,6,7,8,9], params[2], marker='+', linestyle=':', linewidth=1, label='CIFAR10')
plt.plot([1,2,3,4,5,6,7,8,9], params[3], marker='x', linestyle=':', linewidth=1, label='SVHN')

plt.tight_layout()

plt.xticks(fontsize=20)
plt.savefig('params.png')


plt.figure(dpi=1000)
plt.ylabel('H(ε)', fontsize=20)
plt.xlabel('ε', fontsize=20)
plt.xticks([1,2,3,4,5,6,7,8,9],ticks)
plt.ylim(0, 1)
plt.plot([1,2,3,4,5,6,7,8,9], topology[0], marker='.', linestyle=':', linewidth=1, label='MNIST')
plt.plot([1,2,3,4,5,6,7,8,9], topology[1], marker='1', linestyle=':', linewidth=1, label='FMNIST')
plt.plot([1,2,3,4,5,6,7,8,9], topology[2], marker='+', linestyle=':', linewidth=1, label='CIFAR10')
plt.plot([1,2,3,4,5,6,7,8,9], topology[3], marker='x', linestyle=':', linewidth=1, label='SVHN')

plt.tight_layout()

plt.xticks(fontsize=20)
plt.savefig('topology.png')
