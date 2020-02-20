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

To be added soon

## How to cite TFGP
Authors of scientific papers including results generated using TFGP should cite the following paper:

```xml
@article{DEAP_JMLR2012, 
    author    = " F\'elix-Antoine Fortin and Fran\c{c}ois-Michel {De Rainville} and Marc-Andr\'e Gardner and Marc Parizeau and Christian Gagn\'e ",
    title     = { {DEAP}: Evolutionary Algorithms Made Easy },
    pages    = { 2171--2175 },
    volume    = { 13 },
    month     = { jul },
    year      = { 2012 },
    journal   = { Journal of Machine Learning Research }
}
```
