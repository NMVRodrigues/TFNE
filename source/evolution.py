from encoding import *
from random import randint, random
import math
import tensorflow as tf
from copy import deepcopy
from apply_mutations import topology_mutations, parameter_mutations, learning_mutations
from saving_handler import *
import os


class Evolution:
    def __init__(self, population_size, tournament_size, generations, data_information):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.data_information = data_information

    def evolve(self, runs):
        for run in range(runs):

            losses, accuracies, losses_test, accuracies_test = [], [], [], []

            population = self.generate_population(self.population_size)
            for generation in range(self.generations):

                parents = tournament(population, self.population_size, self.tournament_size)
                population, mutations = self.apply_operators(parents)
                #population = self.elitism(offspring, best)

                print("Loss: ", population[0].loss)
                print("Loss test: ", population[0].test_loss)
                print("Accuracy: ", population[0].acc)
                print("Accuracy test: ", population[0].test_acc, '\n')

                losses.append(population[0].loss)
                losses_test.append(population[0].test_loss)
                accuracies.append(population[0].acc)
                accuracies_test.append(population[0].test_acc)

                save_genome(str(population[0]), generation, run)

            save_evolution(losses, accuracies, losses_test, accuracies_test, run)


    def generate_population(self, size):
        individuals = []
        for _ in range(size):
            genotype = Genome()
            genotype.create(params, self.data_information['shape'], self.data_information['nclasses'])
            phenotype = genotype.phenotype()
            history = phenotype.fit(self.data_information['train_data'], epochs=self.data_information['epochs'],
                                    steps_per_epoch=math.ceil(self.data_information['nt_examples'] /
                                                            self.data_information['batch']))
            test_loss, test_accuracy = phenotype.evaluate(self.data_information['test_data'],
                                                        steps=math.ceil(self.data_information['nT_examples'] /
                                                                        self.data_information['batch']))
            del phenotype

            genotype.acc = history.history['accuracy'][-1]
            genotype.loss = history.history['loss'][-1]
            genotype.test_acc = test_accuracy
            genotype.test_loss = test_loss
            
            individuals.append(genotype)

        return sorted(individuals, key=lambda x: x.loss)



    def apply_operators(self, parents):
        mutations = []
        offspring = []
        for parent in parents:
            indiv_copy = deepcopy(parent)
            mutation, indiv_copy = topology_mutations(indiv_copy)
            mutations.append(mutation)
            indiv_copy.reset_values()

            phenotype = indiv_copy.phenotype()
            history = phenotype.fit(self.data_information['train_data'], epochs=self.data_information['epochs'],
                                    steps_per_epoch=math.ceil(self.data_information['nt_examples'] /
                                                            self.data_information['batch']))
            test_loss, test_accuracy = phenotype.evaluate(self.data_information['test_data'],
                                                        steps=math.ceil(self.data_information['nT_examples'] /
                                                                        self.data_information['batch']))
            del phenotype

            indiv_copy.acc = history.history['accuracy'][-1]
            indiv_copy.loss = history.history['loss'][-1]
            indiv_copy.test_acc = test_accuracy
            indiv_copy.test_loss = test_loss
            
            offspring.append(indiv_copy)

        return sorted(offspring, key=lambda x:  x.loss), mutations



    def elitism(self, offspring, parent):
        offspring.append(parent)
        newgen = sorted(offspring, key=lambda x: x[1])  # sorting by loss, increasing
        return newgen[:-1]


def tournament(parents, popsize, tsize):
    chosen = []
    append = chosen.append
    while len(chosen) < popsize:
        r = [randint(0, popsize-1) for x in range(0, tsize)]
        append(parents[min(r)])

    chosen = sorted(chosen, key=lambda x:  x.loss)

    return chosen
