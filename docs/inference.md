# Inference
Inference is the process of making predictions using an estimator. You can think of an estimator as a function that maps unknown samples to predictions by *inferring* their output given the input and the estimator's hidden state (called a *model*) created during training. Estimators are organized into their own namespaces by type. Some meta-estimators such as [Pipeline](pipeline.md) and [Grid Search](grid-search.md) are polymorphic i.e. they can become a classifier, or a regressor, or an anomaly detector based on the type of the estimator(s) they wrap.

## Making Predictions
The most basic output of an estimator is a prediction. All estimators that implement the [Estimator](estimator.md) interface have the `predict()` method which takes a dataset of unknown samples and returns their predictions in order. There are 4 types of estimators in Rubix ML and each type has a different output for a prediction as shown in the table below.

| Estimator Type | Prediction | Examples |
|---|---|---|
| Classifier | A categorical class label | `cat`, `dog`, `ship` |
| Regressor | A continuous value | `490,000` or `1.67592` |
| Clusterer | A discrete cluster number | `0`, `1`, `2`, etc. |
| Anomaly Detector | `1` for an anomaly, `0` otherwise | `0` or `1` |

**Example**

```php
// Import dataset of unkown samples

$predictions = $estimator->predict($dataset);

var_dump($predictions);
```

```sh
array(3) {
  [0]=>
  string(7) "cat"
  [1]=>
  string(8) "dog"
  [2]=>
  string(7) "frog"
}
```

## Estimation of Probabilities

On the map ...

## Ranking Samples

On the map ...