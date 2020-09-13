# Choosing an Estimator
Estimators make up the core of the Rubix ML library and include classifiers, regressors, clusterers, anomaly detectors, and meta-estimators organized into their own namespaces. They are responsible for making predictions and are usually trained with data. Most estimators allow tuning by adjusting their user-defined hyper-parameters. Hyper-parameters are arguments to the learning algorithm that effect its behavior during training and inference. The values for the hyper-parameters can be chosen by intuition, [tuning](choosing-an-esimator.md#hyper-parameter-tuning), [optimization](automl.md#hyper-parameter-optimization), or completely at random. The defaults provided by the library are a good place to start for most problems. To instantiate a new estimator, pass the desired values of the hyper-parameters to the estimator's constructor like in the example below.

```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KNearestNeighbors(10, false, new Minkowski(2.5));
```

## Classifiers
Classifiers are supervised learners that predict a categorical *class* label. They can be used to recognize (`cat`, `dog`, `turtle`), differentiate (`spam`, `not spam`), or describe (`running`, `walking`) the samples in a dataset based on the labels it was trained on. In addition, classifiers that implement the [Probabilistic](probabilistic.md) interface can predict the joint probability distribution of each possible class from the training set.

| Classifier | Flexibility | Proba | Online | Advantages | Disadvantages |
|---|---|---|---|---|---|
| [AdaBoost](classifiers/adaboost.md) | High | ● | | Boosts most classifiers, Learns sample weights | Sensitive to noise, Susceptible to overfitting |
| [Classification Tree](classifiers/classification-tree.md) | Moderate | ● | | Interpretable model, Automatic feature selection | High variance error |
| [Extra Tree Classifier](classifiers/extra-tree-classifier.md) | Moderate | ● | | Faster training, Lower variance error | Similar to Classification Tree |
| [Gaussian Naive Bayes](classifiers/gaussian-nb.md) | Moderate | ● | ● | Requires little data, Highly scalable | Strong Gaussian and feature independence assumption, Sensitive to noise |
| [K-d Neighbors](classifiers/k-d-neighbors.md) | Moderate | ● | | Faster inference | Not compatible with certain distance kernels |
| [K Nearest Neighbors](classifiers/k-nearest-neighbors) | Moderate | ● | ● | Intuitable model, Zero-cost training | Slower inference, Suffers from the curse of dimensionality |
| [Logistic Regression](classifiers/logistic-regression.md) | Low | ● | ● | Interpretable model, Highly Scalable | High bias error, Limited to binary classification |
| [Multilayer Perceptron](classifiers/multilayer-perceptron.md) | High | ● | ● | Handles very high dimensional data, Universal function approximator | High computation and memory cost, Black box |
| [Naive Bayes](classifiers/naive-bayes.md) | Moderate | ● | ● | Requires little data, Highly scalable | Strong feature independence assumption |
| [Radius Neighbors](classifiers/radius-neighbors.md) | Moderate | ● | | Robust to outliers, Quasi-anomaly detector | Radius may be hard to tune, Not guaranteed to return a prediction |
| [Random Forest](classifiers/random-forest.md) | High | ● | | Handles imbalanced datasets, Computes reliable feature importances | High computation and memory cost |
| [Softmax Classifier](classifiers/softmax-classifier.md) | Low | ● | ● | Highly Scalable | High bias error |
| [SVC](classifiers/svc.md) | High | | | Handles high dimensional data | Difficult to tune, Not suitable for large datasets |

## Regressors
Regressors are a type of supervised learner that predict a continuous-valued outcome such as `1.275` or `655`. They can be used to quantify a sample such as its credit score, age, or steering wheel position. Unlike classifiers whose range of predictions is bounded by the number of possible classes in the training set, regressors' range of predictions is unbounded such that the number of possible predictions it could make is infinite.

| Regressor | Flexibility | Online | Verbose | Advantages | Disadvantages |
|---|---|---|---|---|---|
| [Adaline](regressors/adaline.md) | Low | ● | ● | Highly Scalable | High bias error |
| [Extra Tree Regressor](regressors/extra-tree-regressor.md) | Moderate | | | Faster training, Lower variance error | Similar to Regression Tree |
| [Gradient Boost](regressors/gradient-boost.md) | High | | ● | High precision, Computes reliable feature importances | Prone to overfitting, High computation and memory cost |
| [K-d Neighbors Regressor](regressors/k-d-neighbors-regressor.md) | Moderate | | | Faster inference | Not compatible with certain distance kernels |
| [KNN Regressor](regressors/knn-regresor.md) | Moderate | ● | | Intuitable model, Zero-cost training | Slower inference, Suffers from the curse of dimensionality |
| [MLP Regressor](regressors/mlp-regressor.md) | High | ● | ● | Handles very high dimensional data, Universal function approximator | High computation and memory cost, Black box |
| [Radius Neighbors Regressor](regressors/radius-neighbors-regressor.md) | Moderate | | | Robust to outliers, Quasi-anomaly detector | Radius may be hard to tune, Not guaranteed to return a prediction |
| [Regression Tree](regressors/regression-tree.md) | Moderate | | | Interpretable model, Automatic feature selection | High variance error |
| [Ridge](regressors/ridge.md) | Low | | | Interpretable model | High bias error |
| [SVR](regressors/svr.md) | High | | | Handles high dimensional data | Difficult to tune, Not suitable for large datasets |

## Clusterers
Clusterers are unsupervised learners that predict the integer-valued cluster number of the sample such as `0`, `1`, `...`, `n`. They are similar to classifiers, however since they lack a supervised training signal, they cannot be used to recognize or describe samples. Instead, clusterers focus on differentiating and grouping samples using only the patterns discovered in the sample's features. Clusterers that implement the [Probabilistic](probabilistic.md) interface also have the facility to output the probability that a sample belongs to a particular cluster.

| Clusterer | Flexibility | Proba | Online | Advantages | Disadvantages |
|---|---|---|---|---|---|
| [DBSCAN](clusterers/dbscan.md) | High | | | Finds arbitrarily-shaped clusters, Quasi-anomaly detector | Cannot be trained, Slower inference |
| [Fuzzy C Means](clusterers/fuzzy-c-means.md) | Low | ● | | Fast training and inference, Soft clustering | Solution depends highly on initialization, Not suitable for large datasets |
| [Gaussian Mixture](clusterers/gaussian-mixture.md) | Moderate | ● | | Captures irregularly-shaped clusters, Fast training and inference | Strong Gaussian and feature independence assumption |
| [K Means](clusterers/k-means.md) | Low | ● | ● | Highly scalable | Has local minima |
| [Mean Shift](clusterers/mean-shift.md) | Moderate | ● | | Handles non-convex clusters, No local minima | Slower training |

## Anomaly Detectors
Anomaly Detectors are unsupervised learners that predict a boolean-valued outcome encoded as `1` for an outlier or `0` for a regular sample. They are specialized to perform *one class* classification on unbalanced datasets without the need for labeled data. In addition, anomaly detectors that implement the [Ranking](ranking.md) interface can output an anomaly score for each sample in a dataset which can be used to rank the samples from highest to lowest likelihood of being an outlier.

| Anomaly Detector | Scope | Ranking | Online | Advantages | Disadvantages |
|---|---|---|---|---|---|
| [Gaussian MLE](anomaly-detectors/gaussian-mle.md) | Global | ● | ● | Fast training and inference, Highly scalable | Strong Gaussian and feature independence assumption, Sensitive to noise |
| [Isolation Forest](anomaly-detectors/isolation-forest.md) | Local (Features) | ● | | Faster training, Handles high dimensional data | Slower Inference |
| [Local Outlier Factor](anomaly-detectors/local-outlier-factor.md) | Local (Samples) | ● | | Intuitable model, Finds anomalies within clusters | Suffers from the curse of dimensionality |
| [Loda](anomaly-detectors/loda.md) | Local (Features) | ● | ● | Highly scalable | High memory cost |
| [One Class SVM](anomaly-detectors/one-class-svm.md) | Global | | | Handles high dimensional data | Difficult to tune, Not suitable for large datasets |
| [Robust Z-Score](anomaly-detectors/robust-z-score.md) | Global | ● | | Interpretable model, Robust to outliers in the training set | Problems with highly skewed dataset  |

## Model Flexibility
A characteristic of most estimator types is the notion of *flexibility*. Flexibility can be expressed in different ways but greater flexibility usually comes with the capacity to handle more complex tasks. The tradeoff for flexibility is increased computational complexity, reduced interpretability, and greater susceptibility to [overfitting](cross-validation.md#overfitting). In contrast, inflexible models tend to be easier to interpret and quicker to train but are more prone to [underfitting](cross-validation.md#underfitting). In general, we recommend choosing the simplest estimator for your project that does not underfit the training data.

## Meta-estimator Ensembles
Ensemble learning is when multiple estimators are used to make the final prediction on a sample. Meta-estimator Ensembles can consist of multiple variations of the same estimator or a heterogeneous mix of estimators of the same type. They are *polymorphic* in the sense that they take on the type of the base estimators they wrap. They generally work by the principal of averaging and can often achieve greater accuracy than a single estimator.

### Bootstrap Aggregator
Bootstrap Aggregation or *bagging* is an ensemble learning technique that trains learners that each specialize on a unique subset of the training set known as a bootstrap set. The final prediction made by the meta-estimator is the average prediction returned by the ensemble. In the example below, we'll wrap a [Regression Tree](regressors/regression-tree.md) in a [Bootstrap Aggregator](bootstrap-aggregator.md) meta-estimator to form a *forest* of 1000 trees.

```php
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;

$estimator = new BootstrapAggregator(new RegressionTree(5), 1000);
```

### Committee Machine
[Committee Machine](committee-machine.md) is a voting ensemble consisting of estimators (referred to as *experts*) with user-programmable *influence* weights that can be trained in [Parallel](parallel.md). 

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
At some point you may ask yourself "Why do we need so many different learning algorithms?" The answer to that question can be understood by the [No Free Lunch Theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem) which states that, when averaged over the space of *all* possible problems, no learner performs any better than the next. Perhaps a more useful way of stating NFL is that certain learners perform better at certain tasks and worse in others. This is explained by the fact that all learning algorithms have some prior knowledge inherent in them whether it be via the choice of hyper-parameters or the design of the algorithm itself. Another consequence of No Free Lunch is that there exists no single estimator that performs better for all problems.
