# TFNE

TFNE is a Tensorflow based Neuroevolution package. It provides a modular grammar based Neuroevolution algorithm that is very simple to modify and enhance, along with multiple Fitness Landscapes analysis methods.

Full file documentation comming soon.


TF includes the following features:

  * Modular JSON grammar for layers and parameters
  * Genome encoding structure with phenotype decoding and multiple other useful operation
  * Evolution strategies (Currently only GA approach, Grammatical evolution to be added)
  * Metropolis-Hastings sampling
  * Selective walks over the landscape
  * 3 independent mutation operators (Topological, Layer parameters and Learning parameters)
  * Generalization analysis
  * Multiple fitness landscape analysis methods:
    * Autocorrelation
    * Overfitting measure
    * Density Clouds
    * Entropic Measure of Ruggedness
    * Fitness Clouds
    * Negative Slope Coefficient
  * Saving module with function to save network topologies as well as csvs containing data from evolution/samples/walks  


## Requirements

```
Python 3
Tensorflow 2.x
numpy
scipy
matplotlib
seaborn
pandas
json
```
### Optional requirements
These requirements optional.
```
tensorflow_datasets 
```
Makes for testing and running the algorithm easy since there is no need to build a data pipeline. Used in provided examples.

```
Cuda 10.1
CUDNN
```
Both only needed if the user intend to run experiments on GPU. The included examples include the GPU device selection and call, but they can be simply removed if the user wants to run on CPU.

## Example & Notebooks

The following example gives a quick overview of how to run experiments using TFNE,in this case, doing neuroevolution on MNIST.  Full example on run.py, and another example using the overfitting artificial dataset similar to the one described in the paper [citation or link] on runoverfit.py.

```python

"""
##################################################################
Imports and environment settings such as loggers and runtime 
##################################################################
"""
from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
import tensorflow_datasets as tfds
import time
from selective_walk import SelectiveWalk
from evolution import Evolution
from sample import Sample
import logging
import numpy as np

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

start_time = time.time()

"""
##################################################################
Import the dataset and pre process the data 
##################################################################
"""
# Importing MNIST from tf datasets
train_dataset, metadata = tfds.load('mnist:3.*.*', split='train', as_supervised=True, with_info=True)
test_dataset = tfds.load('mnist:3.*.*', split='test', as_supervised=True)

# Getting the number of examples from each partition
num_train_examples = int(metadata.splits['train'].num_examples)
num_test_examples = int(metadata.splits['test'].num_examples)

# Getting the number of classes
num_classes = metadata.features['label'].num_classes

# Normalizing the pizel values to be in the [0,1] range
def normalize(images, labels):
    print(labels)
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset =  train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# Defining a data pipeline to be fed into the model
BATCH_SIZE = 64
NUM_EPOCHS = 8
train_dataset = train_dataset.cache().shuffle(num_train_examples).batch(BATCH_SIZE).repeat(NUM_EPOCHS)
test_dataset = test_dataset.cache().shuffle(num_test_examples).batch(BATCH_SIZE).repeat(1)

"""
##################################################################
Create a data dictionary that will be passed to the algorithms
##################################################################
"""

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

"""
##################################################################
Call and run the desired algorithm
##################################################################
"""
# Creating a neuroevolution object with the following parameters:
# population of 7 individuals with a tournament size 2 and 20 generations
evo = Evolution(7,2,20,data_information) 
# Calling the GPU
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        # Performing 10 neuroevolution runs
        evo.evolve(10)


print("--- %s seconds ---" % (time.time() - start_time))
```
In the following example we show how simple it is to add new layers to the grammar, byt adding a batch normalization layer.

```xml
{
    "Conv": {
        "filters": [32,64,128,256],
        "kernel_size": [2,3,4,5],
        "stride": [1,2,3],
        "activation": ["relu", "elu", "sigmoid"],
        "use_bias": [true, false]
    },
    "Pool": {
        "type": ["Max", "Avg"],
        "pool_size": [2,3,4,5],
        "stride": [1,2,3]
    },
    "Dense": {
        "units": [8,16,32,64,128,256,512],
        "activation": ["relu", "elu", "sigmoid"],
        "use_bias": [true, false]
    },
    "Drop": {
        "active": [true, false]
    },
    "BatchNorm": {
        "momentum": [0.999, 0.99, 0.9],
        "epsilon": [0.01, 0.001, 0.0001]
    },
    "Optimizer": {
        "lr": [0.01, 0.001, 0.0001, 0.00001],
        "decay": [0.01, 0.001, 0.0001, 0.00001],
        "momentum": [0.99, 0.9, 0.5, 0.1],
        "nesterov": [true, false]
    }
}
```

## How to run
Simply clone the repo and run the given examples or make new scripts in the same format.


## How to cite TFNE
Papers that include results generated using TFNE should cite the following paper:

```xml
To Be Added 
```
