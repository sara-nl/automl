# Overview of automated machine learning tools

# Introduction

We want to create an internal overview of current state of the art in machine learning automation tools. Our objective is to characterize this field by possible tasks and existing solutions.

We have in scope automated scaling. This is common in the cloud environment, however that is only partially in this scope. We focus on the HPC environment.

Finally, we should reach conclusions with regard to present bottlenecks and make recommendations about future development.

# Possible tasks

We present two very popular tasks, hyperparameter tuning and Neural Architecture Search (NAS) and describe also two less prevalent tasks, automatic data transformations and automatic scaling.

## Hyperparameter tuning

The problem of the automated machine learning in the past used to be solved by optimizing hyperparameters. This solution assumed that we have a chosen algorithm A with parameter α located in the area ϴ where the target corresponds to finding the value of α parameter found in ϴ and minimizing the function [l (α) = L (A α)](https://ml.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf). Another option that proved to be effective was the method where instead of one single parameter, many model hyperparameters were selected synchronously, [using the gradient of the model selection](https://ieeexplore.ieee.org/document/857804). More advanced approach is represented by a Bayesian method of optimizing hyperparameters; it is based on the Gaussian process operating correctly on small numerical hyperparameters , [SMAC based on random forests and Tree Parzen Estimator](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf), which are suitable for optimizing the high-dimensional hyperparameters in deep neural networks. The aim of the Bayesian optimization is to compose a probabilistic model and then use it to decide how the next function shall be evaluated. The principle is to apply the information gathered from previous estimation, [which enables locating the minimum of non-convex complicated function to optimize hyperparameters](https://ieeexplore.ieee.org/document/8803605).

## Neural Architecture Search

The main idea behind a neural architecture search is to [automatically design a neural network](https://arxiv.org/pdf/2006.02903.pdf) architecture as best as possible without any intervention of machine learning experts. Searching for a well-performed architecture could be power- and time-consuming, whereas many recent papers utilized only a limited number of hyperparameters (number of layers, nodes, activation function and optimizing algorithm). [Functional modules can be aggregated to a larger complex network](https://ieeexplore.ieee.org/document/8793329). Three types of optimization algorithm for searching the problem space are dominantly present. 
* Reinforcement learning. Applying the [reinforcement learning](https://www.sciencedirect.com/science/article/abs/pii/S0262885619300885) able to generate convolutional neural network architecture from scratch by using the gradient method. Accuracy is awarded and then used to compute the gradient to update the controller. This controller is autoregressive. That means, it sets the new hyperparameters at the same time. Another approach is not based on starting with “guessing” and composing a network from scratch. Instead, it can use one of the pre-trained network architecture to achieve the top result. 
* Genetic algorithms. Alternatively, genetic algorithms might be applied. Genetic algorithms are heuristically searching strategies used to solve optimization problems inspired by evolutionary theory. In this case, the optimal neural network structure is found by scoring each one, over the value of network accuracy hyperparameter. The top rank received by the network is processed to set up new generation until maximum generations are achieved. The generated network architecture trained by algorithm is called a child network.
* Bayesian optimization. Apply [Bayesian optimization](https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf) in order to find the value that minimizes an objective function by building a probability model based on past results. This new function is cheaper to optimize than the objective function. Based on successful past results we can guide the optimization process and minimize the expensive evaluations of the objective function. [This can be done for finding new architectures](https://arxiv.org/abs/1910.11858). This is generally [based on a Gaussian process](https://papers.nips.cc/paper/7472-neural-architecture-search-with-bayesian-optimisation-and-optimal-transport.pdf). There are new approaches that focus [on a 1-shot (1 epoch) learning scheme](http://proceedings.mlr.press/v97/zhou19e/zhou19e.pdf).

## Automatic data transformations
Present approaches
* [Semantic data augmentation](https://papers.nips.cc/paper/9426-implicit-semantic-data-augmentation-for-deep-networks.pdf)
* [Invertible](https://openreview.net/pdf?id=HJsjkMb0Z) data augmentation for adversarial training
* [Geometric](https://arxiv.org/pdf/1902.04615.pdf) data augmentation and [group equivariance](https://staff.fnwi.uva.nl/m.welling/wp-content/uploads/papers/icml2016_GCNN.pdf) 

## Automatic scaling
This is done by the cloud Function-as-a-Service (FaaS) providers. This should also ensure efficient hardware utilization, similar to [EfficientNet](https://arxiv.org/abs/1905.11946) and [AmeobaNet](https://arxiv.org/pdf/1901.01074.pdf).

# Summary of tools
| Candidate       | Hyperparameter tuning | Architecture search                     | Data transformations                      | Actively maintained | Autoscale             | Ease of use | Connects to backend | Tested |
| --------------- | --------------------- | --------------------------------------- | ----------------------------------------- | ------------------- | --------------------- | ----------- | ------------------- | ------ |
| auto-pytorch    | yes                   | based on multiple regular architectures | yes, can be extended                      | ?                   | yes accross nodes (?) | average     | Pytorch             | yes    |
| auto-sklearn    | yes                   | limited to scikit                       | limited to sklearn pipes, can be extended | ?                   | yes within a node     | easy        | scikit-learn        | yes    |
| autokeras       | yes                   | based on Keras layers                   | ?                                         | yes                 | yes within a node     | easy        | Keras               | yes    |
| automated ml    | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| automl          | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| ax              | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| deap            | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| devol           | yes                   | yes                                     |                                           |                     |                       |             |                     | yes    |
| far-ho          | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| harmonica       | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| hyperopt        | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| mpi-learn       | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| nevergrad       | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| nni             | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| optimls         | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| optuna          | yes                   | ?                                       | ?                                         | yes                 | yes accross nodes     | ?           | Multiple            | no     |
| pycma           | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| sagemaker       | yes                   | yes (?)                                 |                                           |                     |                       |             |                     | no     |
| scikit-optimize | yes                   | limited to scikit                       |                                           |                     |                       |             |                     | yes    |
| sherpa          | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| smac3           | yes                   | no (?)                                  | ?                                         | ?                   | yes within a node     | ?           | Multiple            | no     |
| talos           | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| tpot            | yes                   | limited to scikit | xboost              | limited to sklearn pipes, can be extended | ?                   | yes within a node     | easy        | Multiple            | yes    |
| tune            | yes                   |                                         |                                           |                     |                       |             |                     | no     |
| vizier          | yes                   |                                         |                                           |                     |                       |             |                     | no     |

# Tools considered
The Freiburg group that developed automl ([github](https://github.com/automl/Auto-PyTorch) [website](https://www.automl.org/)) offers three of the tools included in this comparison. Their approach relies on Bayesian optimization to perform hyperparameter search and Bayesian and genetic optimization for neural architecture search. They support three of the tools presented here.

## auto-sklearn [website](https://automl.github.io/auto-sklearn/master/) [paper](https://docs.google.com/a/chalearn.org/viewer?a=v&pid=sites&srcid=Y2hhbGVhcm4ub3JnfGF1dG9tbHxneDoyYThjZjhhNzRjMzI3MTg4) [github](https://github.com/automl/auto-sklearn)

auto-sklearn uses the Automl concepts and is based on Auto Weka ([website](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/) [pyautoweka](https://github.com/automl/pyautoweka)) and automates sklearn functions

The main functions are
`**autosklearn.classification.AutoSklearnClassifier**`  - classification
`**autosklearn.regression.AutoSklearnRegressor**` - regression 

## smac3 [github](https://github.com/automl/SMAC3) 

Based on [BOAH](https://github.com/automl/BOAH); alternatives [BOCS](https://github.com/baptistar/BOCS) [HpBandSter](https://github.com/automl/HpBandSter) 
Has a parallel version [psmac](https://automl.github.io/SMAC3/master/psmac.html) which uses file comunication between the pool of tasks

## auto-pytorch [github](https://github.com/automl/Auto-PyTorch) 
## tpot [github](https://github.com/EpistasisLab/tpot) [article](https://towardsdatascience.com/automated-machine-learning-d8568857bda1)
## autokeras [website](https://autokeras.com/) [github](https://github.com/keras-team/autokeras) 
## optuna [github](https://github.com/optuna/optuna) 
## mpi-learn [paper](https://arxiv.org/pdf/1712.05878.pdf)
## hyperopt [github](https://hyperopt.github.io/hyperopt/) [paper1](https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf) [paper2](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
## SageMaker [website](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
## Vizier [paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf) [website](https://cloud.google.com/ai-platform/training/docs/using-hyperparameter-tuning)
## H2O Automl [website](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
## Ray Tune [website](https://docs.ray.io/en/master/)
## Talos [github](https://github.com/autonomio/talos)
## Ax [github](https://github.com/facebook/Ax)
## scikit-optimize [github](https://github.com/scikit-optimize/scikit-optimize)
## far-ho [github](https://github.com/lucfra/FAR-HO)
## deap [github](https://github.com/DEAP/deap)
## devol [github](https://github.com/joeddav/devol)
## sherpa [github](https://github.com/sherpa-ai/sherpa) 
## nni [github](https://github.com/Microsoft/nni)
## pycma [github](https://github.com/CMA-ES/pycma)
## nevergrad [github](https://github.com/facebookresearch/nevergrad)
## harmonica [github](https://github.com/callowbird/Harmonica) 
## optimls [github](https://bigml.com/api/optimls)
## automated ml [website](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-auto-train-models)

Uses [hyperdrive](https://www.microsoft.com/en-us/research/publication/hyperdrive-exploring-hyperparameters-pop-scheduling/) under the hood for hyperparameter tuning.

# Conclusions & Recommendations

Current results show very high computational costs.

# Other Resources & Articles
https://www.sciencedirect.com/science/article/pii/S0933365719310437
https://www.springer.com/gp/book/9783030053178

https://medium.com/thinkgradient/automated-machine-learning-an-overview-5a3595d5c4b5
https://hackernoon.com/a-brief-overview-of-automatic-machine-learning-solutions-automl-2826c7807a2a
https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml

https://en.wikipedia.org/wiki/Automated_machine_learning


https://www.ml4aad.org/automl
https://www.capgemini.com/gb-en/2020/02/automatic-machine-learning

----------

