# Choosing an Estimator
Estimators make up the core of the Rubix ML library and include classifiers, regressors, clusterers, anomaly detectors, and meta-estimators organized into their own namespaces. They are responsible for making predictions and are usually trained with data. Most estimators allow tuning by adjusting their user-defined hyper-parameters. Hyper-parameters are arguments to the learning algorithm that effect its behavior during training and inference. The values for the hyper-parameters can be chosen by intuition, [tuning](hyper-parameter-tuning.md), or completely at random. The defaults provided by the library are a good place to start for most problems. To instantiate a new estimator, pass the desired values of the hyper-parameters to the estimator's constructor like in the example below.

```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KNearestNeighbors(10, false, new Minkowski(2.5));
```

## Classifiers
Classifiers are supervised learners that predict a categorical *class* label. They can be used to recognize (`cat`, `dog`, `turtle`), differentiate (`spam`, `not spam`), or describe (`running`, `walking`) the samples in a dataset based on the labels they were trained on. In addition, classifiers that implement the [Probabilistic](probabilistic.md) interface can infer the joint probability distribution of each possible class given an unclassified sample.

| Name | Flexibility | [Proba](probabilistic.md) | [Online](online.md) | [Ranks Features](ranks-features.md) | [Verbose](verbose.md) | Data Compatibility |
|---|---|---|---|---|---|---|
| [AdaBoost](classifiers/adaboost.md) | High | ● | | | ● | Depends on base learner |
| [Classification Tree](classifiers/classification-tree.md) | Medium | ● | | ● | | Categorical, Continuous |
| [Extra Tree Classifier](classifiers/extra-tree-classifier.md) | Medium | ● | | ● | | Categorical, Continuous |
| [Gaussian Naive Bayes](classifiers/gaussian-naive-bayes.md) | Medium | ● | ● | | | Continuous |
| [K-d Neighbors](classifiers/kd-neighbors.md) | Medium | ● | | | | Depends on distance kernel |
| [K Nearest Neighbors](classifiers/k-nearest-neighbors.md) | Medium | ● | ● | | | Depends on distance kernel |
| [Logistic Regression](classifiers/logistic-regression.md) | Low | ● | ● | ● | ● | Continuous |
| [Multilayer Perceptron](classifiers/multilayer-perceptron.md) | High | ● | ● | | ● | Continuous |
| [Naive Bayes](classifiers/naive-bayes.md) | Medium | ● | ● | | | Categorical |
| [Radius Neighbors](classifiers/radius-neighbors.md) | Medium | ● | | | | Depends on distance kernel |
| [Random Forest](classifiers/random-forest.md) | High | ● | | ● | | Categorical, Continuous |
| [Softmax Classifier](classifiers/softmax-classifier.md) | Low | ● | ● | | ● | Continuous |
| [SVC](classifiers/svc.md) | High | | | | | Continuous |

## Regressors
Regressors are a type of supervised learner that predict a continuous-valued outcome such as `1.275` or `655`. They can be used to quantify a sample such as its credit score, age, or steering wheel position in units of degrees. Unlike classifiers whose range of predictions is bounded by the number of possible classes in the training set, a regressor's range is unbounded - meaning, the number of possible values a regressor *could* predict is infinite.

| Name | Flexibility | [Online](online.md) | [Ranks Features](ranks-features.md) | [Verbose](verbose.md) | [Persistable](persistable.md) | Data Compatibility |
|---|---|---|---|---|---|---|
| [Adaline](regressors/adaline.md) | Low | ● | ● | ● | ● | Continuous |
| [Extra Tree Regressor](regressors/extra-tree-regressor.md) | Medium | | ● | | ● | Categorical, Continuous |
| [Gradient Boost](regressors/gradient-boost.md) | High | | ● | ● | ● | Categorical, Continuous |
| [K-d Neighbors Regressor](regressors/kd-neighbors-regressor.md) | Medium | | | | ● | Depends on distance kernel |
| [KNN Regressor](regressors/knn-regressor.md) | Medium | ● | | | ● | Depends on distance kernel |
| [MLP Regressor](regressors/mlp-regressor.md) | High | ● | | ● | ● | Continuous |
| [Radius Neighbors Regressor](regressors/radius-neighbors-regressor.md) | Medium | | | | ● | Depends on distance kernerl |
| [Regression Tree](regressors/regression-tree.md) | Medium | | ● | | ● | Categorical, Continuous |
| [Ridge](regressors/ridge.md) | Low | | ● | | ● | Continuous |
| [SVR](regressors/svr.md) | High | | | | | Continuous |

## Clusterers
Clusterers are unsupervised learners that predict an integer-valued cluster number such as `0`, `1`, `...`, `n`. They are similar to classifiers, however since they lack a supervised training signal, they cannot be used to recognize or describe samples. Instead, clusterers differentiate and group samples using only the information found within the structure of the samples without their labels.

| Name | Flexibility | [Proba](probabilistic.md) | [Online](online.md) | [Verbose](verbose.md) | [Persistable](persistable.md) | Data Compatibility |
|---|---|---|---|---|---|---|
| [DBSCAN](clusterers/dbscan.md) | High | | | | | Depends on distance kernel |
| [Fuzzy C Means](clusterers/fuzzy-c-means.md) | Low | ● | | ● | ● | Continuous |
| [Gaussian Mixture](clusterers/gaussian-mixture.md) | Medium | ● | | ● | ● | Continuous |
| [K Means](clusterers/k-means.md) | Low | ● | ● | ● | ● | Continuous |
| [Mean Shift](clusterers/mean-shift.md) | Medium | ● | | ● | ● | Continuous |

## Anomaly Detectors
Anomaly Detectors are unsupervised learners that predict whether a sample should be classified as an anomaly or not. We use the value `1` to indicate an outlier and `0` for a regular sample and the predictions can be cast to their boolean equivalent if needed. Anomaly detectors that implement the [Scoring](scoring.md) interface can output an anomaly score that can be used to sort the samples by their degree of anomalousness.

| Name | Scope | [Scoring](scoring.md) | [Online](online.md) | [Verbose](verbose.md) | [Persistable](persistable.md) | Data Compatibility |
|---|---|---|---|---|---|---|
| [Gaussian MLE](anomaly-detectors/gaussian-mle.md) | Global | ● | ● | | ● | Continuous |
| [Isolation Forest](anomaly-detectors/isolation-forest.md) | Local | ● | | | ● | Categorical, Continuous |
| [Local Outlier Factor](anomaly-detectors/local-outlier-factor.md) | Local | ● | | | ● | Depends on distance kernel |
| [Loda](anomaly-detectors/loda.md) | Local | ● | ● | | ● | Continuous |
| [One Class SVM](anomaly-detectors/one-class-svm.md) | Global | | | | ● | Continuous |
| [Robust Z-Score](anomaly-detectors/robust-z-score.md) | Global | ● | | | ● | Continuous  |

## Model Flexibility Tradeoff
A characteristic of most estimator types is the notion of *flexibility*. Flexibility can be expressed in different ways but greater flexibility usually comes with the capacity to handle more complex tasks. The tradeoff for flexibility is increased computational complexity, reduced model interpretability, and greater susceptibility to [overfitting](cross-validation.md#overfitting). In contrast, low flexibility models tend to be easier to interpret and quicker to train but are more prone to [underfitting](cross-validation.md#underfitting). In general, we recommend choosing the simplest model that does not underfit the training data for your project.

## Meta-estimator Ensembles
Ensemble learning is when multiple estimators are used together to make the final prediction of a sample. Meta-estimator ensembles can consist of multiple variations of the same estimator or a heterogeneous mix of estimators of the same type. They generally work by the principal of averaging and can often achieve greater accuracy than a single estimator at the cost of training more models.

### Bootstrap Aggregator
Bootstrap Aggregation or *bagging* is an ensemble learning technique that trains a set of learners that each specialize on a unique subset of the training set known as a bootstrap set. The final prediction made by the meta-estimator is the averaged prediction returned by the ensemble. In the example below, we'll wrap a [Regression Tree](regressors/regression-tree.md) in a [Bootstrap Aggregator](bootstrap-aggregator.md) to form a *forest* of 1000 trees.

```php
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;

$estimator = new BootstrapAggregator(new RegressionTree(5), 1000);
```

### Committee Machine
[Committee Machine](committee-machine.md) is a voting ensemble consisting of estimators (referred to as *experts*) with user-programmable *influences*. Each expert is trained on the same dataset and the final prediction is based on the contribution of each expert weighted by their influence.

```php
use Rubix\ML\CommitteeMachine;
use Rubix\ML\RandomForest;
new Rubix\ML\SoftmaxClassifier;
use Rubix\ML\AdaBoost;
use Rubix\ML\ClassificationTree;
use Rubix\ML\Backends\Amp;

$estimator = new CommitteeMachine([
    new RandomForest(),
    new SoftmaxClassifier(128),
    new AdaBoost(new ClassificationTree(5), 1.0),
], [
    3.0, 1.0, 2.0, // Influences
]);
```

## No Free Lunch Theorem
At some point you may ask yourself "Why do we need so many different learning algorithms?" The answer to that question can be understood by the [No Free Lunch Theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem) which states that, when averaged over the space of *all* possible problems, no algorithm performs any better than the next. Perhaps a more useful way of stating NFL is that certain learners perform better at certain tasks and worse in others. This is explained by the fact that all learning algorithms have some prior knowledge inherent in them whether it be via the choice of hyper-parameters or the design of the algorithm itself. Another consequence of No Free Lunch is that there exists no single estimator that performs better for all problems.
