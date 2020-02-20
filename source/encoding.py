from random import random, randint, choice, getrandbits, uniform
import json
import tensorflow as tf

with open('parameter_values.json') as json_file:
    params = json.load(json_file)

class Genome:
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.n_conv = 0
        self.n_pool = 0
        self.n_dense = 0
        self.n_drop = 0
        self.acc = None
        self.loss = None
        self.test_acc = None
        self.test_loss = None
    
    def create(self, params, i_shape, o_shape):
        split1 = randint(1,5)
        split2 = randint(1,5)

        self.layers.append(Conv(params['Conv'], True, i_shape))
        self.n_conv += 1

        for _ in range(split1-1):
            if randint(0,1) == 0:
                self.layers.append(Conv(params['Conv']))
                self.n_conv += 1
            else:
                self.layers.append(Pool(params['Pool']))
                self.n_pool += 1

        self.layers.append('flatten \n')

        for _ in range(split2-1):
            if randint(0,2) < 2:
                self.layers.append(Dense(params['Dense']))
                self.n_dense += 1
            else:
                self.layers.append(Drop())
                self.n_drop += 1
        
        self.layers.append(Dense(params['Dense'], True, o_shape))
        self.n_dense += 1

        self.optimizer = Optimizer(params['Optimizer'])


    def phenotype(self):

        model = tf.keras.Sequential()
        for layer in self.layers:
            if type(layer) == Conv:
                if layer.input_shape is not None:
                    model.add(tf.keras.layers.Conv2D(
                        filters=layer.filters, kernel_size=layer.kernel_size, strides=layer.stride,
                        padding='same', activation=layer.activation, use_bias=layer.use_bias, input_shape=layer.input_shape))
                else:
                     model.add(tf.keras.layers.Conv2D(
                        filters=layer.filters, kernel_size=layer.kernel_size, strides=layer.stride,
                        padding='same', activation=layer.activation, use_bias=layer.use_bias))
            elif type(layer) == Pool:
                if layer.type == 'Max':
                    model.add(tf.keras.layers.MaxPooling2D(pool_size=layer.pool_size, strides=layer.stride, padding='same'))
                else:
                    model.add(tf.keras.layers.AveragePooling2D(pool_size=layer.pool_size, strides=layer.stride, padding='same'))
            elif type(layer) == Drop:
                model.add(tf.keras.layers.Dropout(layer.rate))
            elif type(layer) == Dense:
                model.add(tf.keras.layers.Dense(units=layer.units, activation=layer.activation, use_bias=layer.use_bias))
            else:
                model.add(tf.keras.layers.Flatten())
        
        model.compile(optimizer=tf.keras.optimizers.SGD(
                            lr=self.optimizer.lr, momentum=self.optimizer.momentum,
                            decay=self.optimizer.decay, nesterov=self.optimizer.nesterov), 
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
                    
        return model

    def count_layers(self):

        self.n_conv, self.n_pool, self.n_dense, self.n_drop = 0,0,0,0

        for layer in self.layers:
            if type(layer) == Conv:
                self.n_conv += 1
            elif type(layer) == Pool:
                self.n_pool += 1
            elif type(layer) == Dense:
                self.n_dense += 1
            elif type(layer) == Drop:
                self.n_drop += 1
            else:
                pass
    
    def reset_values(self):
        self.acc, self.loss, self.test_acc, self.test_loss = None, None, None, None

    def __repr__(self):
        return "".join(str(layer) for layer in self.layers) + str(self.optimizer)


    def __ge__(self, other):
        return self.loss <= other.loss


class Conv:
    def __init__(self, params, input_layer=False, input_shape = None):
        self.filters = choice(params['filters'])
        self.kernel_size = choice(params['kernel_size'])
        self.stride = choice(params['stride'])
        self.activation = choice(params['activation'])
        self.use_bias = choice(params['use_bias'])
        if input_layer:
            self.input_shape = input_shape
        else:
            self.input_shape = None

    def __repr__(self):
        return 'Conv \n < filters: ' + str(self.filters) +\
               ' | kernel_size: ' + str(self.kernel_size) +\
               ' | strides: ' + str(self.stride) +\
               ' | activation: ' + str(self.activation) +\
               ' | use_bias: ' + str(self.use_bias) + ' >\n'


class Pool:
    def __init__(self, params):
        self.type = choice(params['type'])
        self.pool_size = choice(params['pool_size'])
        self.stride = choice(params['stride'])

    def __repr__(self):
        return 'Pool \n < type: ' + self.type +\
               ' | pool_size: ' + str(self.pool_size) +\
               ' | strides: ' + str(self.stride) + ' >\n'


class Dense:
    def __init__(self, params, final=False, output_shape=None):
        if final:
            self.units = output_shape
            self.activation = 'softmax'
            self.use_bias = choice(params['use_bias'])
        else:
            self.units = choice(params['units'])
            self.activation = choice(params['activation'])
            self.use_bias = choice(params['use_bias'])

    def __repr__(self):
        return 'Dense \n < units: ' + str(self.units) +\
               ' | activation: ' + str(self.activation) +\
               ' | use_bias: ' + str(self.use_bias) + ' >\n'


class Drop:
    def __init__(self):
        self.rate = uniform(0, 0.7)

    def __repr__(self):
        return 'Drop \n < rate: ' + str(self.rate) + ' >\n'


class Optimizer:
    def __init__(self, params):
        self.lr = choice(params['lr'])
        self.decay = choice(params['decay'])
        self.momentum = choice(params['momentum'])
        self.nesterov = choice(params['nesterov'])

    def __repr__(self):
        return 'Optimizer \n < learning rate: ' + str(self.lr) +\
               ' | decay: ' + str(self.decay) +\
               ' | momentum: ' + str(self.momentum) +\
               ' | nesterov: ' + str(self.nesterov) + '>\n'
