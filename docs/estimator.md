# Estimator
Estimators form the core of the Rubix system because they are responsible for ouputting information from the model (referred to as *predictions*). An estimator can be a Classifier, Regressor, Clusterer, or Anomaly Detector and the interpretation of their predictions depend on the estimator type.

### Make Predictions
Make predictions on a dataset:
```php
public predict(Dataset $dataset) : array
```

**Example**

```php
$predictions = $estimator->predict($dataset);

var_dump($predictions);
```

**Output**

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

> **Note:** The return value of `predict()` is an array containing the predictions indexed in the order the samples were passed to the estimator.