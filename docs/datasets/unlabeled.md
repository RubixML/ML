<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Unlabeled.php">Source</a></span>

# Unlabeled
Unlabeled datasets can be used to train *unsupervised* Estimators and for feeding data into an Estimator to make predictions.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | samples | | array | A 2-dimensional array consisting of rows of samples and columns with feature values. |
| 2 | validate | true | bool | Should we validate the input? |

### Additional Methods

#### Factories
Build a new unlabeled dataset with validation:
```php
public static build(array $samples = []) : self
```

Build a new unlabeled dataset foregoing validation:
```php
public static quick(array $samples = []) : self
```

Build a dataset with an iterator:
```php
public static fromIterator(iterable $samples) : self
```

**Example**

```php
use Rubix\ML\Datasets\Unlabeled;

// Import samples

$dataset = new Unlabeled($samples, true);  // Using the constructor

$dataset = Unlabeled::build($samples);  // Build a dataset with validation

$dataset = Unlabeled::quick($samples);  // Build a dataset without validation

$dataset = Unlabeled::fromIterator($samples); // From an iterator
```