# Estimator
Estimators consist of [Classifiers](#classifiers), [Regressors](#regressors), [Clusterers](#clusterers), [Embedders](#embedders), and [Anomaly Detectors](#anomaly-detectors) that make *predictions* based on data. Estimators that can be trained with data are called *Learners* and they can either be supervised or unsupervised depending on the task. Estimators can employ methods on top of the basic API by implementing a number of addon interfaces such as [Online](#online), [Probabilistic](#probabilistic), [Persistable](#persistable), and [Verbose](#verbose). The most basic Estimator is one that outputs an array of predictions given a dataset of unknown or testing samples.

> **Note**: The return value of `predict()` is an array containing the predictions indexed in the same order that they were fed into the estimator.

To make predictions from a dataset object:
```php
public predict(Dataset $dataset) : array
```

### Example
```php
$predictions = $estimator->predict($dataset);

var_dump($predictions);
```

**Output:**

```sh
array(3) {
  [0]=>
  string(7) "married"
  [1]=>
  string(8) "divorced"
  [2]=>
  string(7) "married"
}
```