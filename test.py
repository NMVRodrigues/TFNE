from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
import tensorflow_datasets as tfds
import time
from keras.datasets import fashion_mnist, cifar10
import sys
#from SelectiveWalk import SelectiveWalk
from Selective_walk import SelectiveWalk
from Evolution_ import Evolution
import math
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

start_time = time.time()
train_dataset, metadata = tfds.load('svhn_cropped:3.*.*', split='train', as_supervised=True, with_info=True)
test_dataset = tfds.load('svhn_cropped:3.*.*', split='test', as_supervised=True)
#train_dataset, metadata = tfds.load('cifar100:3.*.*', split='train[:30%]', as_supervised=True, with_info=True)
#test_dataset = tfds.load('cifar100:3.*.*', split='test[:30%]', as_supervised=True)

#dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
#train_dataset, test_dataset = dataset['train'], dataset['test']
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

num_classes = metadata.features['label'].num_classes

# metadata.features['image'].shape

#(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# add empty color dimension
#x_train = np.expand_dims(x_train, -1)
#x_test = np.expand_dims(x_test, -1)

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255
#print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

def normalize(images, labels):
    print(labels)
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# The map function applies the normalize function to each element in the train.
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()


BATCH_SIZE = 32
train_dataset = train_dataset.take(num_train_examples).repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
#train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.repeat().batch(BATCH_SIZE)







data_information = {
    'train_data': train_dataset,
    'test_data': test_dataset,
    'nt_examples': num_train_examples,
    'nT_examples': num_test_examples,
    'nclasses': num_classes,
    'shape': metadata.features['image'].shape,
    'epochs': 6,
    'batch': 64
}

#swalk = SelectiveWalk(30, 3, data_information)
#if tf.test.is_gpu_available():
#    with tf.device("GPU:0"):
#        swalk.walk(10)


evo = Evolution(7,2,20,data_information)
if tf.test.is_gpu_available():
    with tf.device("GPU:1"):
        evo.evolve(10)

print("--- %s seconds ---" % (time.time() - start_time))


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=3, activation='relu', use_bias=True, input_shape=metadata.features['image'].shape),
#     tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=1, activation='relu', use_bias=False),
#     tf.keras.layers.MaxPooling2D(pool_size=3, strides=1),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0),
#     tf.keras.layers.Dense(512, activation='elu', use_bias=False),
#     tf.keras.layers.Dense(10, activation='softmax', use_bias=True)
# ])
#
# model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9,
#                                    decay=0.01, nesterov=True),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_dataset, epochs=6, steps_per_epoch=math.ceil((num_train_examples*90)/100 / 64))