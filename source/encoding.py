from random import random, randint, choice, getrandbits, uniform
import json
import tensorflow as tf

with open('source/parameter_values.json') as json_file:
    params = json.load(json_file)
with open('source/parameter_values_fixed.json') as json_file:
    paramsF = json.load(json_file)

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
        split1 = randint(1,3)
        split2 = randint(1,3)

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

        self.optimizer = Optimizer(params['Optimizer']['SGD'])

    def create_fixed(self, params, encoding, i_shape, o_shape):

        self.layers.append(Conv_fixed(params['Conv'], encoding, True, i_shape))
        self.layers.append(Pool_fixed(params['Pool']))
        self.layers.append(Conv_fixed(params['Conv'], encoding))
        self.layers.append(Pool_fixed(params['Pool']))
        self.layers.append('flatten \n')
        self.layers.append(Dense_fixed(params['Dense'], encoding))
        self.layers.append(Drop_fixed(params['Drop']))
        self.layers.append(Dense_fixed(params['Dense'], encoding, True, o_shape))
        self.optimizer = Optimizer_fixed(params['Optimizer'], encoding)

        self.n_conv += 2
        self.n_pool += 2
        self.n_dense += 2
        self.n_drop += 1

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
    
    def phenotype_fixed(self):

        model = tf.keras.Sequential()
        for layer in self.layers:
            if type(layer) == Conv_fixed:
                if layer.input_shape is not None:
                    model.add(tf.keras.layers.Conv2D(
                        filters=layer.filters, kernel_size=layer.kernel_size, strides=layer.stride,
                        padding='same', activation=layer.activation, use_bias=layer.use_bias, input_shape=layer.input_shape))
                else:
                     model.add(tf.keras.layers.Conv2D(
                        filters=layer.filters, kernel_size=layer.kernel_size, strides=layer.stride,
                        padding='same', activation=layer.activation, use_bias=layer.use_bias))
            elif type(layer) == Pool_fixed:
                if layer.type == 'Max':
                    model.add(tf.keras.layers.MaxPooling2D(pool_size=layer.pool_size, strides=layer.stride, padding='same'))
                else:
                    model.add(tf.keras.layers.AveragePooling2D(pool_size=layer.pool_size, strides=layer.stride, padding='same'))
            elif type(layer) == Drop_fixed:
                model.add(tf.keras.layers.Dropout(layer.rate))
            elif type(layer) == Dense_fixed:
                model.add(tf.keras.layers.Dense(units=layer.units, activation=layer.activation, use_bias=layer.use_bias))
            else:
                model.add(tf.keras.layers.Flatten())
        

        if self.optimizer.type == 'SGD':
            model.compile(optimizer=tf.keras.optimizers.SGD(), 
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(), 
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


class Conv_fixed:
    def __init__(self, params, encoding, input_layer=False, input_shape = None):
        self.filters = params['filters'][encoding[0]]
        self.kernel_size = params['kernel_size']
        self.stride = params['stride']
        self.activation = params['activation'][encoding[1]]
        self.use_bias = params['use_bias']
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


class Pool_fixed:
    def __init__(self, params):
        self.type = params['type']
        self.pool_size = params['pool_size']
        self.stride = params['stride']

    def __repr__(self):
        return 'Pool \n < type: ' + self.type +\
               ' | pool_size: ' + str(self.pool_size) +\
               ' | strides: ' + str(self.stride) + ' >\n'


class Dense_fixed:
    def __init__(self, params, encoding, final=False, output_shape=None):
        if final:
            self.units = output_shape
            self.activation = 'softmax'
            self.use_bias = params['use_bias']
        else:
            self.units = params['units'][encoding[2]]
            self.activation = params['activation'][encoding[3]]
            self.use_bias = params['use_bias']

    def __repr__(self):
        return 'Dense \n < units: ' + str(self.units) +\
               ' | activation: ' + str(self.activation) +\
               ' | use_bias: ' + str(self.use_bias) + ' >\n'


class Drop_fixed:
    def __init__(self, params):
        self.rate = params['value']

    def __repr__(self):
        return 'Drop \n < rate: ' + str(self.rate) + ' >\n'


class Optimizer_fixed:
    def __init__(self, params, encoding):
        self.type = params['type'][encoding[4]]

    def __repr__(self):
        return 'Optimizer \n < type: ' + str(self.type) + ' >\n'
