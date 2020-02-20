from encoding import *
from random import randint
import scipy.stats
import numpy as np
import math
import sys
import tensorflow as tf
from copy import deepcopy
from apply_mutations import topology_mutations, parameter_mutations, learning_mutations
from saving_handler import *



# TODO -> por safety measures caso loss seja nan
class Sample:
    def __init__(self, sampling_type, size, data_information, neighbors=0):
        self.size = size
        self.sampling_type = sampling_type
        self.neighbors = neighbors
        self.data_information = data_information





    def metropolis_hastings(self, restore=False):
        steps = 0

        lower = 0
        upper = 1
        mu = 0.5
        sigma = 0.1

        source = Genome()
        source.create(params, self.data_information['shape'], self.data_information['nclasses'])
        source_phenotype = source.phenotype()
        history = source_phenotype.fit(self.data_information['train_data'], epochs=self.data_information['epochs'],
                                steps_per_epoch=math.ceil(self.data_information['nt_examples'] /
                                                        self.data_information['batch']))
        test_loss, test_acc = source_phenotype.evaluate(self.data_information['test_data'],
                                                    steps=math.ceil(self.data_information['nT_examples'] /
                                                                    self.data_information['batch']))
        del source_phenotype
        loss, acc = history.history['loss'][-1], history.history['accuracy'][-1]

        save_genome(str(source), '', 0)
        save_sample([acc], [test_acc], [loss], [test_loss])

        pair, mutation = self.mutate(source)

        try:
            df = pd.read_csv('~/Dropbox/sample_pairs.csv')
            new_entry = df.append({'acc':pair.acc, 'acc test':pair.test_acc, 'loss':pair.loss, 'loss test': pair.test_loss, 'mutations':mutation}, ignore_index=True)
            new_entry.to_csv('~/Dropbox/sample_pairs.csv', index=False)
        except:
            save_sample([pair.acc], [pair.test_acc], [pair.loss], [pair.test_loss], True, [mutation])

        save_genome(str(source), '_pair', 0)

        while(steps < self.size-1):

            new_sample = self.generate_neighbor()

            alpha = scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=1)[0]
            print('alpha: ', alpha)
            print('new: ', new_sample.loss)
            print('original: ', loss, '\n')
            
            if (1 / (1+ new_sample.loss)) / (1 / (1+loss)) >= alpha:
                source = new_sample
                loss = new_sample.loss
                acc = new_sample.acc
                test_acc = new_sample.test_acc
                test_loss = new_sample.test_loss
                save_genome(str(new_sample), '', steps)
            else:
                save_genome(str(source), '', steps)
            
            df = pd.read_csv('~/Dropbox/sample.csv')
            new_entry = df.append({'acc':acc, 'acc test':test_acc, 'loss':loss, 'loss test': test_loss}, ignore_index=True)
            new_entry.to_csv('~/Dropbox/sample.csv', index=False)

            pair, mutation = self.mutate(source)
            df = pd.read_csv('~/Dropbox/sample_pairs.csv')
            new_entry = df.append({'acc':pair.acc, 'acc test':pair.test_acc, 'loss':pair.loss, 'loss test':pair.test_loss, 'mutations':mutation}, ignore_index=True)
            new_entry.to_csv('~/Dropbox/sample_pairs.csv', index=False)
            save_genome(str(pair), '_pair', steps)

            steps += 1
            
            tf.random.set_seed(1)


    def generate_neighbor(self):
        genotype = Genome()
        genotype.create(params, self.data_information['shape'], self.data_information['nclasses'])
        phenotype = genotype.phenotype()
        history = phenotype.fit(self.data_information['train_data'], epochs=self.data_information['epochs'],
                                steps_per_epoch=math.ceil(self.data_information['nt_examples'] /
                                                        self.data_information['batch']))
        genotype.test_loss, genotype.test_acc = phenotype.evaluate(self.data_information['test_data'],
                                                    steps=math.ceil(self.data_information['nT_examples'] /
                                                                    self.data_information['batch']))
        del phenotype
        genotype.loss, genotype.acc = history.history['loss'][-1], history.history['accuracy'][-1]
        
        return genotype

    
    def mutate(self, parent):
        mutations = []
        neighbors = []

        for _ in range(self.neighbors):
            indiv_copy = deepcopy(parent)
            mutation, indiv_copy = parameter_mutations(indiv_copy)
            mutations.append(mutation)
            indiv_copy.reset_values()

            phenotype = indiv_copy.phenotype()
            history = phenotype.fit(self.data_information['train_data'], epochs=self.data_information['epochs'],
                                    steps_per_epoch=math.ceil(self.data_information['nt_examples'] /
                                                            self.data_information['batch']))
            indiv_copy.test_loss, indiv_copy.test_acc = phenotype.evaluate(self.data_information['test_data'],
                                                        steps=math.ceil(self.data_information['nT_examples'] /
                                                                        self.data_information['batch']))
            del phenotype

            indiv_copy.acc, indiv_copy.loss = history.history['accuracy'][-1], history.history['loss'][-1]
            
            neighbors.append(indiv_copy)

        best = min(neighbors, key=lambda x: x.loss)
        return best, mutations[neighbors.index(best)]