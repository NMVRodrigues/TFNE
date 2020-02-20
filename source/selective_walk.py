from encoding import *
from random import randint
import math
import tensorflow as tf
from copy import deepcopy
from apply_mutations import topology_mutations, parameter_mutations, learning_mutations
from saving_handler import *



class SelectiveWalk:
    def __init__(self, size, neighbors, data_information):
        self.size = size
        self.neighbors = neighbors
        self.data_information = data_information

    def walk(self, runs):

        #mirrored_strategy = tf.distribute.MirroredStrategy()
        #with mirrored_strategy.scope():

        for run in range(runs):
            print('going for walk', '\n')
            source = Genome()
            source.create(params, self.data_information['shape'], self.data_information['nclasses'])
            source_phenotype = source.phenotype()
            #with mirrored_strategy.scope():
            history = source_phenotype.fit(self.data_information['train_data'], epochs=self.data_information['epochs'],
                                    steps_per_epoch=math.ceil(self.data_information['nt_examples'] /
                                                            self.data_information['batch']))
            test_loss, test_accuracy = source_phenotype.evaluate(self.data_information['test_data'],
                                                        steps=math.ceil(self.data_information['nT_examples'] /
                                                                        self.data_information['batch']))
            del source_phenotype
            losses, accuracies = [history.history['loss'][-1]], [history.history['accuracy'][-1]]
            losses_test, accuracies_test = [test_loss], [test_accuracy]
            mutations = ['-']
            n = 1

            save_genome(str(source), 0, run)

            while n < self.size:

                neighbors, _mutations = create_neighbors(source, self.neighbors)
                for neighbor in neighbors:
                    # with mirrored_strategy.scope():
                    ph = neighbor.phenotype()
                    history = ph.fit(self.data_information['train_data'],
                                            epochs=self.data_information['epochs'],
                                            steps_per_epoch=math.ceil(self.data_information['nt_examples'] /
                                                                      self.data_information['batch']))
                    test_loss, test_accuracy = ph.evaluate(self.data_information['test_data'],
                                                           steps=math.ceil(self.data_information['nT_examples'] /
                                                                           self.data_information['batch']))
                    del ph
                    neighbor.loss = history.history['loss'][-1]
                    neighbor.acc = history.history['accuracy'][-1]
                    neighbor.test_acc = test_accuracy
                    neighbor.test_loss = test_loss

                source = min(neighbors, key=lambda x: x.loss)

                losses.append(source.loss)
                accuracies.append(source.acc)
                losses_test.append(source.test_loss)
                accuracies_test.append(source.test_acc)
                mutations.append(_mutations[neighbors.index(source)])

                del neighbors, _mutations
                tf.random.set_seed(1)

                save_genome(str(source), n, run)

                n += 1

            save_walk(accuracies, accuracies_test, losses, losses_test, mutations, run)

            tf.random.set_seed(1)


def create_neighbors(source, number):
    mutations = []
    individuals = []

    for _ in range(number):
        indiv_copy = deepcopy(source)
        mutation, indiv_copy = topology_mutations(indiv_copy)
        indiv_copy.reset_values()
        mutations.append(mutation)
        individuals.append(indiv_copy)
    return individuals, mutations
