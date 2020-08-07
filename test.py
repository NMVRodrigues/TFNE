from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
import tensorflow_datasets as tfds
import time
import math
from source.selective_walk import SelectiveWalk
from source.evolution import Evolution
from source.sample import Sample
import logging
import numpy as np
from source.encoding import *

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
NUM_EPOCHS = 10
train_dataset = train_dataset.cache().shuffle(num_train_examples).batch(BATCH_SIZE).repeat(NUM_EPOCHS)
test_dataset = test_dataset.cache().shuffle(num_test_examples).batch(BATCH_SIZE).repeat(1)


data_information = {
    'train_data': train_dataset,
    'test_data': test_dataset,
    'nt_examples': num_train_examples,
    'nT_examples': num_test_examples,
    'nclasses': num_classes,
    'shape': metadata.features['image'].shape,
    'epochs': NUM_EPOCHS,
    'batch': BATCH_SIZE
}

if tf.test.is_gpu_available():
    genotype = Genome()
    genotype.create_fixed(paramsF, [3,2,2,2,0], data_information['shape'], data_information['nclasses'])
    phenotype = genotype.phenotype_fixed()
    phenotype.summary()
    history = phenotype.fit(data_information['train_data'], epochs=data_information['epochs'],
                            steps_per_epoch=math.ceil(data_information['nt_examples'] /
                                                      data_information['batch']))
    test_loss, test_accuracy = phenotype.evaluate(data_information['test_data'],
                                                steps=math.ceil(data_information['nT_examples'] /
                                                                data_information['batch']))

print("--- %s seconds ---" % (time.time() - start_time))

