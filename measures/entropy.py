from scipy import stats, mean, median
import numpy as np
import sys
import pandas as pd
import os
import math
from SavingHandler import save_entropy
import matplotlib.pyplot as plt


####################################################

#
#
# Application of Entropic Measure of Ruggedness as described by Vassilev and K. Malan
#
#

####################################################









csvspath = 'samplepath'


accs = []
taccs = []
loss = []
tloss = []


for i in range(10):
    df = pd.read_csv(os.path.join(csvspath, 'run_'+str(i)+'.csv'), sep=';')
    accs.append(np.array(df['acc'])), taccs.append(np.array(df['acc test']))
    loss.append(np.array(df['loss'])), tloss.append(np.array(df['loss test']))


entropies = [[],[],[],[],[],[],[],[],[]]

for x in range(len(accs)):
    accs[x] = np.append(accs[x], accs[x][0])


for walk in accs:
    stability = max(np.absolute(np.diff(walk)))
    epsilons = [0., stability/128, stability/64, stability/32, stability/16, stability/8, stability/4, stability/2, stability]
    for index,epsilon in enumerate(epsilons):
        pairs = {
            '01': 0,
            '0\u03C4': 0,
            '10': 0,
            '1\u03C4': 0,
            '\u03C40': 0,
            '\u03C41': 0
        }
        print('\u03B5: ', epsilon)
        String = []
        for i in range(1, len(walk)):
            if walk[i] - walk[i-1] < -1 * epsilon:
                String.append('\u03C4')
            elif abs(walk[i] - walk[i-1]) <= epsilon:
                String.append('0')
            else:
                String.append('1')
        
        print(String)
        String_ = ''.join(String)
        n = len(String) - 1
        result = 0
        for elem, next_val in zip(*[iter(String)] * 2):
            if elem != next_val:
                pair = elem + next_val
                pairs[pair] += 1
            else:
                pass
        for pair in pairs.items():
            if pair[1] != 0:
                result += (pair[1]/n) * (math.log(pair[1]/n, 6))
            else:
                pass
        result *= -1
        entropies[index].append(result)

        print('H(\u03B5) = ', result, '\n')

avg = [mean(epsilon) for epsilon in entropies]

for index,epsilon in enumerate(entropies):
    epsilon.append(-1)
    epsilon.append(avg[index])


save_entropy(entropies, maindir)