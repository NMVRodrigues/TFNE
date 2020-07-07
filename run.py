from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
import tensorflow_datasets as tfds
import time
from source.selective_walk import SelectiveWalk
from source.evolution import Evolution
from source.sample import Sample
import logging
import numpy as np

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

start_time = time.time()

train_dataset, metadata = tfds.load('mnist:3.*.*', split='train', as_supervised=True, with_info=True)
test_dataset = tfds.load('mnist:3.*.*', split='test', as_supervised=True)


num_train_examples = int(metadata.splits['train'].num_examples)
num_test_examples = int(metadata.splits['test'].num_examples)

num_classes = metadata.features['label'].num_classes

def normalize(images, labels):
    print(labels)
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset =  train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

#.take(num_train_examples)
BATCH_SIZE = 64
NUM_EPOCHS = 8
train_dataset = train_dataset.cache().shuffle(num_train_examples).batch(BATCH_SIZE).repeat(NUM_EPOCHS)
test_dataset = test_dataset.cache().shuffle(num_test_examples).batch(BATCH_SIZE).repeat(1)


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

#swalk = SelectiveWalk(30, 3, data_information, ['learning'])
#if tf.test.is_gpu_available():
#    with tf.device("GPU:0"):
#        swalk.walk(10)

evo = Evolution(7,2,20,data_information, ['learning'])
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        evo.evolve(10)

#sample = Sample('', 200, data_information, ['learning'], 3)
#if tf.test.is_gpu_available():
#    with tf.device("GPU:0"):
#        sample.metropolis_hastings()

print("--- %s seconds ---" % (time.time() - start_time))

