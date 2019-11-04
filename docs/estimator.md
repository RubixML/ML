# Estimator
Estimators make up the core of the Rubix ML library because they are responsible for computing predictions from a model (referred to as *inference*). An estimator can be a Classifier, Regressor, Clusterer, or Anomaly Detector and the interpretation of their output depends on the estimator type. For example, the output of a regressor is a single number whereas the predictions made by a classifier will be 1 of k discrete class labels.

### Estimator Outputs
| Estimator Type | Prediction | Example |
|---|---|---|
| Classifier | A categorical class label | 'cat', 'dog' |
| Regressor | A continuous value | 490,000 or 1.67592 |
| Clusterer | A discrete cluster number | '0', '1', '2', etc. |
| Anomaly Detector | '1' for an anomaly, '0' otherwise | '0' or '1' |

### Make Predictions
Return the predictions from a dataset in an array:
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