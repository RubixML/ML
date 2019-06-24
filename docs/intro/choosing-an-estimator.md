
# Choosing an Estimator
Estimators make up the core of the Rubix library as they are responsible for making predictions. There are many different estimators to choose from and each one operates differently. Choosing the right [Estimator](#estimators) for the job is crucial to creating a performant system.

For our simple example we will focus on an easily intuitable classifier called [K Nearest Neighbors](#k-nearest-neighbors). Since the label of each training sample we collect will be a discrete class (*married couples* or *divorced couples*), we need an Estimator that is designed to output class predictions. The K Nearest Neighbors classifier works by locating the closest training samples to an unknown sample and choosing the class label that appears most often.

> **Note**: In practice, you will test out a number of different estimators to get the best sense of what works for your particular dataset.

## Create the Estimator Instance
Like most estimators, the K Nearest Neighbors (KNN) classifier requires a set of parameters (called *hyper-parameters*) to be chosen up front by the user. These parameters control how the learner behaves during training and inference. These parameters can be selected based on some prior knowledge of the problem space, or at random. The defaults provided in Rubix are a good place to start for most machine learning problems.

In K Nearest Neighbors, the hyper-parameter *k* is the number of nearest points from the training set to compare an unknown sample to in order to infer its class label. For example, if the 5 closest neighbors to a given unknown sample have 4 married labels and 1 divorced label, then the algorithm will output a prediction of married with a probability of 0.8.

The second hyper-parameter is the distance *kernel* that determines how distance is measured within the model. We'll go with standard [Euclidean](#euclidean) distance for now.

Then, to instantiate the K Nearest Neighbors classifier ...

```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KNearestNeighbors(5, new Euclidean());
```

> **Note**: You can find a full description of all of the K Nearest Neighbors hyper-parameters in the [API reference](#k-nearest-neighbors).