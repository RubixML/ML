<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Datasets/Unlabeled.php">[source]</a></span>

# Unlabeled
Unlabeled datasets are used to train unsupervised learners and for feeding unknown samples into an estimator to make predictions. As their name implies, they do not require a corresponding label for each sample.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | samples | | array | A 2-dimensional array consisting of rows of samples and columns with feature values. |
| 2 | verify | true | bool | Should we verify the data? |

## Example

```php
use Rubix\ML\Datasets\Unlabeled;

$samples = [
    [0.1, 20, 'furry'],
    [2.0, -5, 'rough'],
    [0.001, -10, 'rough'],
];

$dataset = new Unlabeled($samples);
```

## Additional Methods
This dataset does not have any additional methods.
