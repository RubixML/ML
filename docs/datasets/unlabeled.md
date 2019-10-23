<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Unlabeled.php">Source</a></span>

# Unlabeled
Unlabeled datasets are used to train *unsupervised* learners and for feeding data into an estimator to make predictions during inference.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | samples | | array | A 2-dimensional array consisting of rows of samples and columns with feature values. |
| 2 | validate | true | bool | Should we validate the input? |

### Additional Methods

#### Factory Methods
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

$samples = [
    [0.1, 20, 'furry'],
    [2.0, -5, 'rough'],
    [0.001, -10, 'rough'],
];

$dataset = new Unlabeled($samples, true);

$dataset = Unlabeled::build($samples);  // With validation

$dataset = Unlabeled::quick($samples);  // Without validation

$dataset = Unlabeled::fromIterator($samples); // From an iterator
```