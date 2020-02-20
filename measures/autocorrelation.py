from scipy import stats, mean, median
import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr

csvspath = 'samplepath'

accs = []
taccs = []
loss = []
tloss = []

ACaccs = []
ACtaccs = []
ACloss = []
ACtloss = []

values = [[], [], [], []]

linesep = pd.Series([np.nan])

for i in range(10):
    df = pd.read_csv(os.path.join(csvspath, 'run_'+str(i)+'.csv'), sep=';')
    accs.append(df['acc']), taccs.append(df['acc test'])
    loss.append(df['loss']), tloss.append(df['loss test'])


for i in range(10):
    _values = [['acc'], ['test acc'], ['loss'], ['test loss']]
    for step in range(1, 5):
        step_1_l = loss[i][:-step]
        step_2_l = loss[i][step:]

        step_1_a = accs[i][:-step]
        step_2_a = accs[i][step:]

        step_1_L = tloss[i][:-step]
        step_2_L = loss[i][step:]

        step_1_A = taccs[i][:-step]
        step_2_A = taccs[i][step:]

        corelation_l = pearsonr(step_1_l, step_2_l)[0]
        corelation_a = pearsonr(step_1_a, step_2_a)[0]

        corelation_L = pearsonr(step_1_L, step_2_L)[0]
        corelation_A = pearsonr(step_1_A, step_2_A)[0]

        _values[0].append(corelation_a)
        _values[1].append(corelation_A)
        _values[2].append(corelation_l)
        _values[3].append(corelation_L)

    values[0].append(_values[0])
    values[1].append(_values[1])
    values[2].append(_values[2])
    values[3].append(_values[3])

values = [pd.DataFrame(entry, columns=['-','step 1', 'step 2', 'step 3', 'step 4']) for entry in values]
values = pd.concat(values).to_csv(csvspath + 'autocorrelations.csv', sep=";", index=False)


