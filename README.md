# TFNE

TFNE is a Tensorflow based Neuroevolution package. It provides a modular grammar based Neuroevolution algorithm that is very simple to modify and enhance, along with multiple Fitness Landscapes analysis methods.


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

BATCH_SIZE = 64
NUM_EPOCHS = 8
train_dataset = train_dataset.cache().shuffle(num_train_examples).batch(BATCH_SIZE).repeat(NUM_EPOCHS)
test_dataset = test_dataset.cache().shuffle(num_test_examples).batch(BATCH_SIZE).repeat(1)


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


evo = Evolution(7,2,20,data_information)
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        evo.evolve(10)


print("--- %s seconds ---" % (time.time() - start_time))
```

## How to cite TFGP
Papers that include results generated using TFGP should cite the following paper:

```xml
To Be Added IEE ACCESS bibtex ref
```
