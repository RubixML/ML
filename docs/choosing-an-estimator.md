# Choosing an Estimator
Estimators make up the core of the Rubix ML library and include classifiers, regressors, clusterers, anomaly detectors, and meta-estimators organized into their own namespaces. They are responsible for making predictions and can often be trained with data. Some meta-estimators such as [Pipeline](pipeline.md) and [Grid Search](grid-search.md) are *polymorphic* i.e. they bear the type of the base estimator that they wrap. Most estimators allow tuning by adjusting their hyper-parameters. To instantiate a new estimator, pass the desired values of the hyper-parameters the estimator's constructor like in the example below.

**Example**

```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Minkowski;

$estimator = new KNearestNeighbors(10, false, new Minkowski(2.0));
```

It is important to note that not all estimators are created equal and choosing the right estimator for your project is important for achieving the best results. In the following sections, we'll break down the estimators available to you in Rubix ML and point out some of their advantages and disadvantages.

## Classifiers
Classifiers can often be graded on their ability to form decision boundaries between the sample clusters that represent the classes in a classification problem. Simple linear classifiers such as Logistic Regression can only handle clusters that are *linearly seperable* whereas highly flexible models such as Multilayer Perceptron are called *universal function approximators* because they can theoretically handle any decision boundary. The tradeoff for flexibility is usually reduced interpretability, increased computational complexity, and susceptibility to [overfitting](cross-validation.md#overfitting). Below we list some of the classifiers in Rubix ML which range from simple interpretable models to highly complex black boxes.

| Classifier | Advantages | Disadvantages |
|---|---|---|
| [AdaBoost](classifiers/adaboost.md) | High precision, Learns influences of base estimators, Boosts most base classifiers | Sensitive to noise, Prone to overfitting, Cannot be parallelized |
| [Classification Tree](classifiers/classification-tree.md) | Interpretable model, automatic feature selection, handles both categorical and continuous data types, handles missing data | High variance, prone to overfitting |
| [Extra Tree Classifier](classifiers/extra-tree-classifier.md) | Faster growth and lower variance compared to Classification Tree | Similar to Classification Tree |
| [Gaussian Naive Bayes](classifiers/gaussian-nb.md) | Quick to compute and update, Online, Highly scalable | Strong assumption on feature independence, Sensitive to noisy features |
| [K-d Neighbors](classifiers/k-d-neighbors.md) | Faster inference speed compared to KNN | Not compatible with categorical distance kernels, cannot be partially trained |
| [K Nearest Neighbors](classifiers/k-nearest-neighbors) | Simple tuning, Intuitable model, Non-parametric, Zero-cost training, Can handle missing data | Slow inference, Requires features to be on the same scale, Suffers from the curse of dimensionality |
| [Logistic Regression](classifiers/logistic-regression.md) | Interpretable model, Very efficient, Online, Scalable | Cannot handle non-linear decision boundaries, Prone to underfitting, Only supports 2 classes |
| [Multilayer Perceptron](classifiers/multilayer-perceptron.md) | Highly flexible model, Online, Universal function approximator | Requires a lot of data, High computation and memory cost, Black box interpretability |
| [Naive Bayes](classifiers/naive-bayes.md) | Fast training and inference, Requires little data, Highly scalable | Strong assumption on feature independence |
| [Radius Neighbors](classifiers/radius-neighbors.md) | More robust to outliers than KNN, quasi-anomaly detector | More difficult tuning compared to KNN, Not guaranteed to return a prediction |
| [Random Forest](classifiers/random-forest.md) | Moderately flexible, Moderate to low variance, Hard to overfit, Parallelizable, Computes reliable feature importances | High computation and memory cost, Less interpretable than a single tree |
| [Softmax Classifier](classifiers/softmax-classifier.md) | Probabilistic interpretation, Online, Multiclass, Scalable | Cannot handle non-linear decision boundaries, prone to underfitting |
| [SVC](classifiers/svc.md) | Works well in high dimensions, Fast inference, Works well with small datasets | Difficult tuning, Not suitable for large datasets |

## Regressors
In much the same way that flexibility is sometimes required to separate data of different classes, greater flexibility can often be required to fit a regression line to a complex hyperplane. Linear models such as Ridge tend to [underfit](cross-validation.md#underfitting) data that is non-linear while more flexible models like Gradient Boost can sometimes overfit the training data if not properly tuned. In general, it's a good idea to choose the simplest regressor that doesn't underfit the dataset.

| Regressor | Advantages | Disadvantages |
|---|---|---|
| [Adaline](regressors/adaline.md) | Fast training and inference, Interpretable model, Online, Highly Scalable | Cannot handle non-linear data, Prone to underfitting |
| [Extra Tree Regressor](regressors/extra-tree-regressor.md) | Faster growth and lower variance compared to Regression Tree | Similar to Regression Tree |
| [Gradient Boost](regressors/gradient-boost.md) | High precision, Flexible model, Works well with both categorical and continuous data, Handles missing data, Outputs feature importances | Prone to overfitting, High computation and memory cost, Cannot be parallelized |
| [K-d Neighbors Regressor](regressors/k-d-neighbors-regressor.md) | Faster inference speed compared to KNN | Not compatible with categorical distance kernels, cannot be partially trained |
| [KNN Regressor](regressors/knn-regresor.md) | Simple tuning, Intuitable model, Non-parametric, Zero-cost training, Can handle missing data | Slow inference, Requires features to be on the same scale, Suffers from the curse of dimensionality |
| [MLP Regressor](regressors/mlp-regressor.md) | Highly flexible model, Online, Universal function approximator | Requires a lot of data, High computation and memory cost, Black box interpretability |
| [Radius Neighbors Regressor](regressors/radius-neighbors-regressor.md) | More robust to outliers than KNN, quasi-anomaly detector | More difficult tuning compared to KNN, Not guaranteed to return a prediction |
| [Regression Tree](regressors/regression-tree.md) | Interpretable model, automatic feature selection, handles both categorical and continuous data types, handles missing data | High variance, prone to overfitting |
| [Ridge](regressors/ridge.md) | Interpretable model, Optional regularization | Cannot handle non-linear data, Prone to underfitting, Cannot be partially trained |
| [SVR](regressors/svr.md) | Works well in high dimensions, Fast inference | Difficult tuning |

## Clusterers
Clusterers can be rated by their ability to represent the outer hull surrounding the samples in a clustering. Simple centroid-based models establish a uniform hypersphere around the points. More flexible clusterers such as Gaussian Mixture can better conform to the hull of the cluster by allowing the surface of the sphere to be irregular and *bumpy*. The tradeoff for flexibility results in more model parameters and with it increased computational complexity.

| Clusterer | Advantages | Disadvantages |
|---|---|---|
| [DBSCAN](clusterers/dbscan.md) | Highly flexible, Capable of non-linear shaped clusters | Cannot be trained, Slow inference speed, Difficult tuning |
| [Fuzzy C Means](clusterers/fuzzy-c-means.md) | Fast inference, Soft clustering, Probabilistic interpretation | Higher memory cost than K Means, Not suitable for large datasets |
| [Gaussian Mixture](clusterers/gaussian-mixture.md) | Simple tuning, Moderately flexible model, Captures non-spherical clusters | Higher memory cost |
| [K Means](clusterers/k-means.md) | Fast training and inference, Online, Highly scalable | Clusters must be linearly separable |
| [Mean Shift](clusterers/mean-shift.md) | Handles non-linear shaped clusters | Slow training speed, Difficult to tune |

## Anomaly Detectors

On the map ...


## Meta-estimators

On the map ...