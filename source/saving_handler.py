import pandas as pd
import os


def save_walk_(accuracies, accuracies_test, losses, losses_test, mutations,
              corelation_l, corelation_L, corelation_a, corelation_A, index, type=''):

    padding = len(losses)

    save = pd.DataFrame({'acc': accuracies, 'acc test': accuracies_test, 'loss': losses, 'loss test': losses_test,
                         'mutes': mutations,
                         'corelation loss': [corelation_l]+['-']*(padding-1),
                         'corelation test loss': [corelation_L]+['-']*(padding-1),
                         'corelation acc': [corelation_a]+['-']*(padding-1),
                         'corelation test acc': [corelation_A]+['-']*(padding-1)},
                        columns=['acc', 'acc test', 'loss', 'loss test', 'mutes',
                                 'corelation loss', 'corelation acc', 'corelation test loss', 'corelation test acc'])
    save.to_csv('~/Dropbox/run_' + str(index) + type + '.csv', sep=';', index=False)

def save_walk(accuracies, accuracies_test, losses, losses_test, mutations, index, type=''):

    padding = len(losses)

    save = pd.DataFrame({'acc': accuracies, 'acc test': accuracies_test,
                         'loss': losses, 'loss test': losses_test, 'mutes': mutations,},
                        columns=['acc', 'acc test', 'loss', 'loss test', 'mutes'])
    save.to_csv('~/Dropbox/run_' + str(index) + type + '.csv', sep=';', index=False)

def save_sample(accuracies, accuracies_test, losses, losses_test, neighbor=False, mutation=''):
    if neighbor is False:
        save = pd.DataFrame({'acc': accuracies, 'acc test': accuracies_test,
                            'loss': losses, 'loss test': losses_test},
                            columns=['acc', 'acc test', 'loss', 'loss test'])
        save.to_csv('~/Dropbox/sample.csv', sep=',', index=False)
    else:
        save = pd.DataFrame({'acc': accuracies, 'acc test': accuracies_test,
                         'loss': losses, 'loss test': losses_test, 'mutations':mutation},
                        columns=['acc', 'acc test', 'loss', 'loss test', 'mutations'])
        save.to_csv('~/Dropbox/sample_pairs.csv', sep=',', index=False)


def save_evolution(losses, accuracies, losses_test, accuracies_test, index):
    save = pd.DataFrame({'acc': accuracies, 'acc test': accuracies_test, 'loss': losses, 'loss test': losses_test},
                        columns=['acc', 'acc test', 'loss', 'loss test'])
    save.to_csv('~/Dropbox/run_' + str(index) + '.csv', sep=';', index=False)


def save_genome(genome, index, run, type='_'):
    with open('data' + str(run) + '_' + str(index) + type + '.txt', 'w') as out:
        genes = str(genome).split('\n')
        for gene in genes:
            out.write(gene + '\n')


def save_entropy(values, maindir):
    save = pd.DataFrame({'0': values[0], '\u03B5*/128': values[1], '\u03B5*/64': values[2], '\u03B5*/32': values[3],
                         '\u03B5*/16': values[4], '\u03B5*/8': values[5], '\u03B5*/4': values[6], '\u03B5*/2': values[7],
                         '\u03B5*': values[8]},
                        columns=['0', '\u03B5*/128', '\u03B5*/64', '\u03B5*/32', '\u03B5*/16', '\u03B5*/8',
                                 '\u03B5*/4', '\u03B5*/2', '\u03B5*'])

    save.to_csv(maindir + 'Entropy.csv', sep=';', encoding='utf-8', index=False)

