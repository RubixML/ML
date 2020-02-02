# Estimator
The Estimator interface is implemented by all learners in Rubix ML. It provides basic inference functionality through the `predict()` method which returns a set of predictions from a dataset. Additionally, it provides methods for returning estimator type and data type compatibility declarations.

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

> **Note:** The return value of `predict()` is an array containing the predictions in the same order that they were indexed in the dataset.

### Estimator Type
Return the integer-encoded estimator type:
```php
public type() : int
```

**Example**

```php
use Rubix\ML\Estimator;

$type = $estimator->type();

var_dump($type); // Dump integer-encoded type

var_dump(Estimator::TYPE_STRINGS[$type]); // Dump human readable type
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
var_dump($estimator->compatibility());
```

```sh
array(2) {
  [0]=>
  object(Rubix\ML\DataType)#20257 (1) {
    ["type":protected]=> int(2)
  }
  [1]=>
  object(Rubix\ML\DataType)#20265 (1) {
    ["type":protected]=> int(1)
  }
}
```

### Hyper-parameters
Return the settings of the hyper-parameters in an associative array:
```php
public params() : array
```

**Example**

```php
var_dump($estimator->params());
```

```sh
array(4) {
  ["max_depth"]=> int(10)
  ["max_leaf_size"]=> int(2)
  ["max_features"]=> int(3)
  ["min_purity_increase"]=> float(1.0E-7)
}
```