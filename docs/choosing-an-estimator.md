# Choosing an Estimator
Estimators make up the core of the Rubix ML library and include classifiers, regressors, clusterers, anomaly detectors, and meta-estimators organized into their own namespaces. They are responsible for making predictions and are usually trained with data. Most estimators allow tuning by adjusting their user-defined hyper-parameters. To instantiate a new estimator, pass the desired values of the hyper-parameters to the estimator's constructor like in the example below.

```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KNearestNeighbors(10, false, new Minkowski(2.0));
```

Choosing the right estimator for your project will help achieve the best results. A common theme among all estimator types is the notion of model *flexibility*. How flexibility is expressed depends on the estimator type but greater flexibility usually includes the capacity to handle more difficult tasks. The tradeoff for increased flexibility is often increased computational complexity, reduced interpretability, and greater susceptibility to [overfitting](cross-validation.md#overfitting). For most use cases, we recommend selecting the simplest model that does not [underfit](cross-validation.md#underfitting) your dataset. In the next sections, we'll break down the estimators available to you in Rubix ML by type, grade their flexibility, and describe some of their advantages and disadvantages.

## Classifiers
Classifiers can be assessed by their ability to form decision boundaries around the areas that define the classes. Simple linear classifiers such as [Logistic Regression](classifiers/logistic-regresion.md) can only handle classes that are *linearly separable*. On the other hand, highly flexible models such as the [Multilayer Perceptron](classifiers/multilayer-perceptron.md) can theoretically learn any decision boundary.

| Classifier | Flexibility | Proba | Online | Advantages | Disadvantages |
|---|---|---|---|---|---|
| [AdaBoost](classifiers/adaboost.md) | High | ● | | Boosts most classifiers, Learns sample weights | Sensitive to noise, Susceptible to overfitting |
| [Classification Tree](classifiers/classification-tree.md) | Moderate | ● | | Interpretable model, Automatic feature selection | High variance |
| [Extra Tree Classifier](classifiers/extra-tree-classifier.md) | Moderate | ● | | Faster training, Lower variance | Similar to Classification Tree |
| [Gaussian Naive Bayes](classifiers/gaussian-nb.md) | Moderate | ● | ● | Requires little data, Highly scalable | Strong Gaussian and feature independence assumption, Sensitive to noise |
| [K-d Neighbors](classifiers/k-d-neighbors.md) | Moderate | ● | | Faster inference | Not compatible with certain distance kernels |
| [K Nearest Neighbors](classifiers/k-nearest-neighbors) | Moderate | ● | ● | Intuitable model, Zero-cost training | Slower inference, Suffers from the curse of dimensionality |
| [Logistic Regression](classifiers/logistic-regression.md) | Low | ● | ● | Interpretable model, Highly Scalable | Prone to underfitting, Limited to binary classification |
| [Multilayer Perceptron](classifiers/multilayer-perceptron.md) | High | ● | ● | Handles very high dimensional data, Universal function approximator | High computation and memory cost, Black box |
| [Naive Bayes](classifiers/naive-bayes.md) | Moderate | ● | ● | Requires little data, Highly scalable | Strong feature independence assumption |
| [Radius Neighbors](classifiers/radius-neighbors.md) | Moderate | ● | | Robust to outliers, Quasi-anomaly detector | Not guaranteed to return a prediction |
| [Random Forest](classifiers/random-forest.md) | High | ● | | Handles imbalanced datasets, Computes reliable feature importances | High computation and memory cost |
| [Softmax Classifier](classifiers/softmax-classifier.md) | Low | ● | ● | Highly Scalable | Prone to underfitting |
| [SVC](classifiers/svc.md) | High | | | Handles high dimensional data | Difficult to tune, Not suitable for large datasets |

## Regressors
In regression, flexibility is expressed in the ability of a regressor to model the process that generated the outcomes of the training data. Simple models such as [Ridge](regressors/ridge.md) assume a linear relationship between input and outcome and tend to underfit data that is complex and nonlinear. More flexible models such as [Gradient Boost](regressors/gradient-boost.md) can model complex processes but are more prone to overfitting if not tuned properly.

| Regressor | Flexibility | Online | Verbose | Advantages | Disadvantages |
|---|---|---|---|---|---|
| [Adaline](regressors/adaline.md) | Low | ● | ● | Highly Scalable | Prone to underfitting |
| [Extra Tree Regressor](regressors/extra-tree-regressor.md) | Moderate | | | Faster training, Lower variance | Similar to Regression Tree |
| [Gradient Boost](regressors/gradient-boost.md) | High | | ● | High precision, Computes reliable feature importances | Prone to overfitting, High computation and memory cost |
| [K-d Neighbors Regressor](regressors/k-d-neighbors-regressor.md) | Moderate | | | Faster inference | Not compatible with certain distance kernels |
| [KNN Regressor](regressors/knn-regresor.md) | Moderate | ● | | Intuitable model, Zero-cost training | Slower inference, Suffers from the curse of dimensionality |
| [MLP Regressor](regressors/mlp-regressor.md) | High | ● | ● | Handles very high dimensional data, Universal function approximator | High computation and memory cost, Black box |
| [Radius Neighbors Regressor](regressors/radius-neighbors-regressor.md) | Moderate | | | Robust to outliers, Quasi-anomaly detector | Not guaranteed to return a prediction |
| [Regression Tree](regressors/regression-tree.md) | Moderate | | | Interpretable model, Automatic feature selection | High variance |
| [Ridge](regressors/ridge.md) | Low | | | Interpretable model | Prone to underfitting |
| [SVR](regressors/svr.md) | High | | | Handles high dimensional data | Difficult to tune, Not suitable for large datasets |

## Clusterers
Clusterers express flexibility by their ability to represent an outer hull surrounding the samples in a cluster. *Hard* clustering algorithms such as [K Means](clusterers/k-means.md) establish a uniform hypersphere around the clusters. This is great for clusters that are linearly separable, however, breaks down when clusters become more interspersed. More flexible models such as [DBSCAN](clusterers/dbscan.md) can better conform to the shape of the cluster by allowing the surface of the hull to be irregular and *bumpy*.

| Clusterer | Flexibility | Proba | Online | Advantages | Disadvantages |
|---|---|---|---|---|---|
| [DBSCAN](clusterers/dbscan.md) | High | | | Finds arbitrarily-shaped clusters, Quasi-anomaly detector | Cannot be trained, Slower inference |
| [Fuzzy C Means](clusterers/fuzzy-c-means.md) | Low | ● | | Fast training and inference, Soft clustering | Solution depends highly on initialization, Not suitable for large datasets |
| [Gaussian Mixture](clusterers/gaussian-mixture.md) | Moderate | ● | | Captures irregularly-shaped clusters, Fast training and inference | Strong Gaussian and feature independence assumption |
| [K Means](clusterers/k-means.md) | Low | ● | ● | Highly scalable | Has local minima |
| [Mean Shift](clusterers/mean-shift.md) | Moderate | ● | | Handles non-convex clusters, No local minima | Slower training |

## Anomaly Detectors
Anomaly Detectors fall into one of two groups - there are those that consider the entire training data when determining an anomaly, and there are those that only consider a *local region* of the training set. Local anomaly detectors are typically more accurate but come with higher computational complexity. Global anomaly detectors are more suited for real-time applications but may produce a higher number of false positives and/or negatives.

| Anomaly Detector | Scope | Ranking | Online | Advantages | Disadvantages |
|---|---|---|---|---|---|
| [Gaussian MLE](anomaly-detectors/gaussian-mle.md) | Global | ● | ● | Fast training and inference, Highly scalable | Strong Gaussian and feature independence assumption, Sensitive to noise |
| [Isolation Forest](anomaly-detectors/isolation-forest.md) | Local | ● | | Fast training, Handles high dimensional data | Slower Inference |
| [Local Outlier Factor](anomaly-detectors/local-outlier-factor.md) | Local | ● | | Intuitable model, Finds anomalies within clusters | Suffers from the curse of dimensionality |
| [Loda](anomaly-detectors/loda.md) | Global | ● | ● | Highly scalable | High memory cost |
| [One Class SVM](anomaly-detectors/one-class-svm.md) | Global | | | Handles high dimensional data | Difficult to tune, Not suitable for large datasets |
| [Robust Z-Score](anomaly-detectors/robust-z-score.md) | Global | ● | | Interpretable model, Robust to outliers in the training set | Problems with highly skewed dataset  |

## Meta-estimators
Meta-estimators wrap and enhance other estimators with extra functionality. They are polymorphic in the sense that they take on the type of the base estimator they wrap. A characteristic feature of meta-estimators that implement the [Wrapper](wrapper.md) interface is that they allow methods to be called on the base estimator by calling them from the meta-estimator.

| Meta-estimator | Usage | Parallel | Verbose | Compatibility |
|---|---|---|---|---|
| [Bootstrap Aggregator](bootstrap-aggregator.md) | Ensemble | ● | | Classifiers, Regressors, Anomaly Detectors |
| [Committee Machine](committee-machine.md) | Ensemble | ● | ● | Classifiers, Regressors, Anomaly Detectors |
| [Grid Search](grid-search.md) | Model Selection | ● | ● | Any |
| [Persistent Model](persistent-model.md) | Model Persistence | | | Any persistable estimator |
| [Pipeline](pipeline.md) | Preprocessing | | ● | Any |

In the example below, we'll wrap a [Regression Tree](regressors/regression-tree.md) in a Bootstrap Aggregator meta-estimator to create a *forest* of 1000 trees.

```php
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;

$estimator = new BootstrapAggregator(new RegressionTree(4), 1000);
```

## No Free Lunch Theorem
At some point you may ask yourself "Why do we need so many different learning algorithms?" The answer to that question can be understood by the *No Free Lunch* (NFL) theorem which states that, when averaged over the space of *all* possible problems, no learner performs any better than the next. Perhaps a more useful way of stating NFL is that certain learners perform better at certain tasks and worse in others. This is explained by the fact that all learning algorithms have some prior knowledge inherent in them whether it be via the choice of hyper-parameters or the design of the algorithm itself. Another consequence of No Free Lunch is that there exists no single estimator that performs better for all problems.
