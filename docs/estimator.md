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
$compatibility = $estimator->compatibility();
```