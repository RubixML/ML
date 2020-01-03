# Choosing an Estimator
Estimators make up the core of the Rubix ML library and include classifiers, regressors, clusterers, anomaly detectors, and meta-estimators organized into their own namespaces. They are responsible for making predictions and are usually trained with data. Some meta-estimators such as [Pipeline](pipeline.md) and [Grid Search](grid-search.md) are *polymorphic* i.e. they bear the type of the base estimator that they wrap. Most estimators allow tuning by adjusting their hyper-parameters. To instantiate a new estimator, pass the desired values of the hyper-parameters to the estimator's constructor like in the example below.

**Example**

```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KNearestNeighbors(10, false, new Minkowski(2.0));
```

It is important to note that not all estimators are created equal and choosing the right estimator for your project is important for achieving the best results. In the following sections, we'll break down the estimators available to you in Rubix ML and point out some of their advantages and disadvantages.

## Classifiers
Classifiers can often be graded on their ability to form decision boundaries between areas that define the classes. Simple linear classifiers such as Logistic Regression can only handle classes that are *linearly separable*. On the other hand, highly flexible models such as the Multilayer Perceptron can theoretically handle any decision boundary. The tradeoff for increased flexibility is reduced interpretability, increased computational complexity, and greater susceptibility to [overfitting](cross-validation.md#overfitting).

| Classifier | Flexibility | Proba | Online | Advantages | Disadvantages |
|---|---|---|---|---|---|
| [AdaBoost](classifiers/adaboost.md) | High | ● | | High precision, Boosts most classifiers | Sensitive to noise, Susceptible to overfitting |
| [Classification Tree](classifiers/classification-tree.md) | Moderate | ● | | Interpretable model, automatic feature selection | High variance, Susceptible to overfitting |
| [Extra Tree Classifier](classifiers/extra-tree-classifier.md) | Moderate | ● | | Fast training, Lower variance | Similar to Classification Tree |
| [Gaussian Naive Bayes](classifiers/gaussian-nb.md) | Moderate | ● | ● | Quick to compute, Requires little data, Highly scalable | Feature independence assumption |
| [K-d Neighbors](classifiers/k-d-neighbors.md) | Moderate | ● | | Fast inference | Not compatible with certain distance kernels |
| [K Nearest Neighbors](classifiers/k-nearest-neighbors) | Moderate | ● | ● | Intuitable model, Zero-cost training | Slow inference, Suffers from the curse of dimensionality |
| [Logistic Regression](classifiers/logistic-regression.md) | Low | ● | ● | Interpretable model, Highly Scalable | Prone to underfitting, Only handles 2 classes |
| [Multilayer Perceptron](classifiers/multilayer-perceptron.md) | High | ● | ● | Universal function approximator | High computation and memory cost, Susceptible to overfitting, Black box |
| [Naive Bayes](classifiers/naive-bayes.md) | Moderate | ● | ● | Quick to compute, Requires little data, Highly scalable | Strong assumption on feature independence |
| [Radius Neighbors](classifiers/radius-neighbors.md) | Moderate | ● | | Robust to outliers, quasi-anomaly detector | Not guaranteed to return a prediction |
| [Random Forest](classifiers/random-forest.md) | High | ● | | Stable, Computes reliable feature importances | High computation and memory cost |
| [Softmax Classifier](classifiers/softmax-classifier.md) | Low | ● | ● | Highly Scalable | Prone to underfitting |
| [SVC](classifiers/svc.md) | High | | | Works well in high dimensions, Fast inference speed | Not suitable for large datasets |

## Regressors
In terms of regression, flexibility is expressed as the ability of a model to fit a regression line to potentially complex non-linear data. Linear models such as Ridge tend to [underfit](cross-validation.md#underfitting) data that is non-linear while more flexible models such as Gradient Boost are prone to overfit the training data if not tuned properly. In general, it's best to choose the simplest regressor that doesn't underfit your dataset.

| Regressor | Flexibility | Online | Advantages | Disadvantages |
|---|---|---|---|---|
| [Adaline](regressors/adaline.md) | Low | ● | Interpretable model, Highly Scalable | Prone to underfitting |
| [Extra Tree Regressor](regressors/extra-tree-regressor.md) | Moderate | | Fast training Lower variance | Similar to Regression Tree |
| [Gradient Boost](regressors/gradient-boost.md) | High | | High precision, Computes reliable feature importances | Prone to overfitting, High computation and memory cost |
| [K-d Neighbors Regressor](regressors/k-d-neighbors-regressor.md) | Moderate | | Fast inference | Not compatible with certain distance kernels |
| [KNN Regressor](regressors/knn-regresor.md) | Moderate | ● | Intuitable model, Zero-cost training | Slow inference, Suffers from the curse of dimensionality |
| [MLP Regressor](regressors/mlp-regressor.md) | High | ● | Universal function approximator | High computation and memory cost, Prone to overfitting, Black box |
| [Radius Neighbors Regressor](regressors/radius-neighbors-regressor.md) | Moderate | | Robust to outliers, quasi-anomaly detector | Not guaranteed to return a prediction |
| [Regression Tree](regressors/regression-tree.md) | Moderate | | Interpretable model, automatic feature selection | High variance, Susceptible to overfitting |
| [Ridge](regressors/ridge.md) | Low | | Interpretable model | Prone to underfitting |
| [SVR](regressors/svr.md) | High | | Works well in high dimensions, Fast inference | Low Precision |

## Clusterers
Clusterers can be rated by their ability to represent the outer hull surrounding the samples in the cluster. Simple centroid-based models such as K Means establish a uniform hypersphere around the clusters. More flexible clusterers such as Gaussian Mixture can better conform to the hull of the cluster by allowing the surface of the hypersphere to be irregular and *bumpy*. The tradeoff for flexibility typically results in more model parameters and with it increased computational complexity.

| Clusterer | Flexibility | Proba | Online | Advantages | Disadvantages |
|---|---|---|---|---|---|
| [DBSCAN](clusterers/dbscan.md) | High | | | Finds arbitrarily-shaped clusters | Cannot be trained, Slow inference |
| [Fuzzy C Means](clusterers/fuzzy-c-means.md) | Low | ● | | Fast inference, Soft clustering | Highly depends on initialization, Not suitable for large datasets |
| [Gaussian Mixture](clusterers/gaussian-mixture.md) | Moderate | ● | | Captures non-spherical clusters | Higher memory cost |
| [K Means](clusterers/k-means.md) | Low | ● | ● | Fast training and inference, Highly scalable | Has local minima, Prone to underfitting |
| [Mean Shift](clusterers/mean-shift.md) | Moderate | ● | | Handles non-convex clusters, No local minima | Slow training |

## Anomaly Detectors

On the map ...


## Meta-estimators
Sometimes, you'll want to enhance your estimator with added functionality such as the ability to save and load from storage. In other cases, you might want to train a bunch of models and average their predictions for greater accuracy. Meta-estimators allow you to wrap nearly any estimator and increase its ability.

| Meta-estimator | Usage | Parallel | Verbose | Wraps |
|---|---|---|---|---|
| [Bootstrap Aggregator](bootstrap-aggregator.md) | Model Ensemble | ● | | Classifiers, Regressors, Anomaly Detectors |
| [Committee Machine](committee-machine.md) | Model Ensemble | ● | ● | Classifiers, Regressors, Anomaly Detectors |
| [Grid Search](grid-search.md) | Model Selection | ● | ● | Any |
| [Persistent Model](persistent-model.md) | Model Persistence | | | Any persistable model |
| [Pipeline](pipelinemd) | Preprocessing | | ● | Any |

In the example below, we'll wrap a Regression Tree in the Bootstrap Aggregator meta-estimator and tell it to train and average the predictions of 1,000 base estimators automatically.

**Example**

```php
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;

$estimator = new BootstrapAggregator(new RegressionTree(4), 1000);
```