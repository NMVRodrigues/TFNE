from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
import tensorflow_datasets as tfds
import time
from Selective_walk import SelectiveWalk
from Evolution_ import Evolution
from sample import Sample
import logging
import numpy as np

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

start_time = time.time()

#train_dataset, metadata = tfds.load('mnist:3.*.*', split='train', as_supervised=True, with_info=True)
#test_dataset = tfds.load('mnist:3.*.*', split='test', as_supervised=True)

train_dataset, metadata = tfds.load('mnist:3.*.*', split='train[70%:]', as_supervised=True, with_info=True)
test_dataset = tfds.load('mnist:3.*.*', split='test[70%:]', as_supervised=True)

num_train_examples = int(metadata.splits['train'].num_examples*0.3)
num_test_examples = int(metadata.splits['test'].num_examples*0.3)

num_classes = metadata.features['label'].num_classes

td_x = np.zeros((3000, 28, 28, 1), dtype=np.uint8)
td_y = np.zeros((3000), dtype=np.uint8)

trd_x = np.zeros((18000, 28, 28, 1), dtype=np.uint8)
trd_y = np.zeros((18000), dtype=np.uint8)


i = 0
one = 0
three = 0
five = 0
seven = 0
nine = 0
for elem in test_dataset:
    if elem[1].numpy() == 1 and one < 200:
        td_x[i] = elem[0]
        td_y[i] = 3
        one += 1
    elif elem[1].numpy() == 3 and three < 200:
        td_x[i] = elem[0]
        td_y[i] = 9
        three += 1
    elif elem[1].numpy() ==5 and five < 200:
        td_x[i] = elem[0]
        td_y[i] = 0
        five += 1
    elif elem[1].numpy() == 7 and seven < 200:
        td_x[i] = elem[0]
        td_y[i] = 4
        seven += 1
    elif elem[1].numpy() == 9 and nine < 200:
        td_x[i] = elem[0]
        td_y[i] = 2
        nine += 1 
    else:
        td_x[i] = elem[0]
        td_y[i] = elem[1].numpy()
    i += 1

i = 0
for elem in train_dataset:
    trd_x[i] = elem[0]
    trd_y[i] = elem[1].numpy()
    i += 1

td_x = tf.data.Dataset.from_tensor_slices(td_x)
td_y = tf.data.Dataset.from_tensor_slices(td_y)
td = tf.data.Dataset.zip((td_x, td_y))

trd_x = tf.data.Dataset.from_tensor_slices(trd_x)
trd_y = tf.data.Dataset.from_tensor_slices(trd_y)
trd = tf.data.Dataset.zip((trd_x, trd_y))

def normalize(images, labels):
    print(labels)
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

del train_dataset
del test_dataset

#rain_dataset =  train_dataset.map(normalize)
td =  td.map(normalize)
trd =  trd.map(normalize)
#test_dataset = test_dataset.map(normalize)

#.take(num_train_examples)
BATCH_SIZE = 64
NUM_EPOCHS = 8
#train_dataset = train_dataset.cache().shuffle(num_train_examples).batch(BATCH_SIZE).repeat(NUM_EPOCHS)
#test_dataset = test_dataset.cache().shuffle(num_test_examples).batch(BATCH_SIZE).repeat(1)
trd = trd.cache().shuffle(num_train_examples).batch(BATCH_SIZE).repeat(NUM_EPOCHS)
td = td.cache().shuffle(num_test_examples).batch(BATCH_SIZE).repeat(1)


data_information = {
    'train_data': trd,
    'test_data': td,
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

#evo = Evolution(7,2,20,data_information)
#if tf.test.is_gpu_available():
#    with tf.device("GPU:0"):
#        evo.evolve(10)

sample = Sample('', 200, data_information, 3)
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        sample.metropolis_hastings()

print("--- %s seconds ---" % (time.time() - start_time))

