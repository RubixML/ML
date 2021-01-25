# Estimator
The Estimator interface is implemented by all learners in Rubix ML. It provides basic inference functionality through the `predict()` method which returns a set of predictions from a dataset. Additionally, it provides methods for returning estimator type and data type compatibility declarations.

### Make Predictions
Return the predictions from a dataset containing unknown samples in an array:
```php
public predict(Dataset $dataset) : array
```

```php
$predictions = $estimator->predict($dataset);

var_dump($predictions);
```

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

!!! note
    The return value of `predict()` is an array containing the predictions in the same order that they were indexed in the dataset.
