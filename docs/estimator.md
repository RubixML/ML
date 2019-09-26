# Estimator
Estimators form the core of Rubix ML because they are responsible for ouputting predictions from the model (referred to as *inference*). An estimator can be a Classifier, Regressor, Clusterer, or Anomaly Detector and the interpretation of their output depends on the estimator type.

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

> **Note:** The return value of `predict()` is an array containing the predictions in the order that they were indexed in the dataset.

### Estimator Type
Return the integer encoded estimator type:
```php
public type() : int
```

**Example**

```php
use Rubix\ML\Estimator;

$type = $estimator->type();

var_dump($type); // Output integer encoded type

var_dump(Estimator::TYPES[$type]); // Output human readable type
```

```sh
int(1)

string(10) "classifier"
```

### Data Type Compatibility
Return the data types that this estimator is compatible with:
```php
public compatibility() : array
```

**Example**

```php
$compatibility = $estimator->compatiility();
```