from .encoding import *
from random import randint, uniform

def add_conv(genome):
    genome.layers.insert(randint(1, genome.n_conv+genome.n_pool), Conv(params['Conv']))
    genome.count_layers()

    return genome


def del_conv(genome):
    selected = randint(1, genome.n_conv-1)
    del_index = 0

    for index, layer in enumerate(genome.layers):
        if type(layer) == Conv:
            if selected == 1:
                del_index = index
            else:
                selected -= 1
        if del_index != 0:
            break
    
    del genome.layers[del_index]

    genome.count_layers()
    return genome

def add_pool(genome):
    genome.layers.insert(randint(1, genome.n_conv+genome.n_pool), Pool(params['Pool']))
    genome.count_layers()
    return genome

def del_pool(genome):
    selected = randint(1, genome.n_pool)
    del_index = 0

    for index, layer in enumerate(genome.layers):
        if type(layer) == Pool:
            if selected == 1:
                del_index = index
            else:
                selected -= 1
        if del_index != 0:
            break
    
    del genome.layers[del_index]

    genome.count_layers()
    return genome

def add_dense(genome):
    genome.layers.insert(randint(genome.n_conv+genome.n_pool + 1, len(genome.layers)-1), Dense(params['Dense']))
    genome.count_layers()
    return genome

def del_dense(genome):
    selected = randint(1, genome.n_dense-1)
    del_index = 0

    for index, layer in enumerate(genome.layers):
        if type(layer) == Dense:
            if selected == 1:
                del_index = index
            else:
                selected -= 1
        if del_index != 0:
            break
    
    del genome.layers[del_index]

    genome.count_layers()
    return genome

def add_drop(genome):
    genome.layers.insert(randint(genome.n_conv+genome.n_pool + 1, len(genome.layers)-1), Drop())
    genome.count_layers()
    return genome

def del_drop(genome):
    selected = randint(1, genome.n_drop)
    del_index = 0

    for index, layer in enumerate(genome.layers):
        if type(layer) == Drop:
            if selected == 1:
                del_index = index
            else:
                selected -= 1
        if del_index != 0:
            break

    del genome.layers[del_index]

    genome.count_layers()
    return genome


# ----------Parameter----------


def change_bias(genome, layer):
    genome.layers[layer].use_bias = choice(params[type(genome.layers[layer]).__name__]['use_bias'])
    return genome


def change_drop(genome, layer):
    genome.layers[layer].rate = uniform(0, 0.7)
    return genome


# hum. possivelmente alterar isso para gravar qual tipo de stride está a ser mutado
def change_stride(genome, layer):
    genome.layers[layer].stride = choice(params[type(genome.layers[layer]).__name__]['stride'])
    return genome


def change_activation(genome, layer):
    #print('str:', str(type(genome.layers[layer])))
    #print('non str:', type(genome.layers[layer]), '\n')
    #print('eq:', type(genome.layers[layer]) is Conv)
    #print('name: ', type(genome.layers[layer]).__name__)
    genome.layers[layer].activation = choice(params[type(genome.layers[layer]).__name__]['activation'])
    return genome


# number of neurons
def change_units(genome, layer):
    genome.layers[layer].units = choice(params['Dense']['units'])
    return genome


# number of kernels
def change_filters(genome, layer):
    genome.layers[layer].filters = choice(params['Conv']['filters'])
    return genome


def change_pool_size(genome, layer):
    genome.layers[layer].pool_size = choice(params['Pool']['pool_size'])
    return genome


def change_kernel_size(genome, layer):
    genome.layers[layer].kernel_size = choice(params['Conv']['kernel_size'])
    return genome



# ----------Learning----------


def change_lr(genome):
    genome.optimizer.lr = choice(params['Optimizer']['SGD']['lr'])
    return genome


def change_decay(genome):
    genome.optimizer.decay = choice(params['Optimizer']['SGD']['decay'])
    return genome


def change_momentum(genome):
    genome.optimizer.momentum = choice(params['Optimizer']['SGD']['momentum'])
    return genome


def change_nesterov(genome):
    genome.optimizer.nesterov = choice(params['Optimizer']['SGD']['nesterov'])
    return genome


# ----------Optimize----------
def change_optimizer(genome):
    pass
