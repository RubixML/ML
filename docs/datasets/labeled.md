<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Datasets/Labeled.php">[source]</a></span>

# Labeled

A Labeled dataset is used to train supervised learners and for testing a model by providing the ground-truth. In addition to the standard dataset API, a labeled dataset can perform operations such as stratification and sorting the dataset using the label column.

!!! note
    Since PHP silently converts integer strings (ex. `'1'`) to integers in some circumstances, you should not use integer strings as class labels. Instead, use an appropriate non-integer string class name such as `'class 1'`, `'#1'`, or `'first'`.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | samples | | array | A 2-dimensional array consisting of rows of samples and columns with feature values. |
| 2 | labels | | array | A 1-dimensional array of labels that correspond to each sample in the dataset. |
| 2 | verify | true | bool | Should we verify the data? |

## Example

```php
use Rubix\ML\Datasets\Labeled;

$samples = [
    [0.1, 20, 'furry'],
    [2.0, -5, 'rough'],
    [0.01, 5, 'furry'],
];

$labels = ['not monster', 'monster', 'not monster'];

$dataset = new Labeled($samples, $labels);
```

## Additional Methods

### Selectors

Return the labels of the dataset in an array.

```php
public labels() : array
```

Return a single label at the given row offset.

```php
public label(int $offset) : mixed
```

Return all of the possible outcomes i.e. the unique labels in an array.

```php
public possibleOutcomes() : array
```

```php
print_r($dataset->possibleOutcomes());
```

```php
Array
(
    [0] => female
    [1] => male
)
```

### Data Types

Return the data type of the label.

```php
public labelType() : Rubix\ML\DataType
```

```php
echo $dataset->labelType();
```

```sh
continuous
```

### Stratification

Group samples by their class label and return them in their own dataset.

```php
public stratifyByClassLabels() : array
```

```php
$strata = $dataset->stratifyByClassLabels();
```

Split the dataset into left and right subsets such that the proportions of class labels remain intact.

```php
public stratifiedSplit($ratio = 0.5) : array
```

```php
[$training, $testing] = $dataset->stratifiedSplit(0.8);
```

Return *k* equal size subsets of the dataset such that class proportions remain intact.

```php
public stratifiedFold($k = 10) : array
```

!!! note
    *k* must be less than or equal to the number of samples in the *smallest* stratum, otherwise an `InvalidArgumentException` is thrown.

```php
$folds = $dataset->stratifiedFold(3);
```

#### Binned Stratification

The methods above group samples by *exact* label equality, which is only meaningful for categorical labels. When the label is continuous, use the binned variants instead. They derive a set of equal frequency bins from the quantiles of the label and then stratify over those bins, which preserves the shape of the target distribution in every subset.

Group samples into equal frequency bins by their continuous label and return them in their own dataset.

```php
public stratifyByLabelBins($bins = 10) : array
```

```php
$strata = $dataset->stratifyByLabelBins(5);
```

Split the dataset into left and right subsets such that the distribution of the continuous label remains intact.

```php
public binnedSplit($ratio = 0.5, $bins = 10) : array
```

```php
[$training, $testing] = $dataset->binnedSplit(0.8);
```

Return *k* equal size subsets of the dataset such that the label distribution remains intact.

```php
public binnedFold($k = 10, $bins = 10) : array
```

```php
$folds = $dataset->binnedFold(5);
```

!!! note
    Unlike the categorical methods, `binnedSplit()` and `binnedFold()` take a bin count. More bins can provide a tighter match to the target distribution, but require enough samples in each split or fold to represent every bin.

### Transform Labels

Transform the labels in the dataset using a callback function and return self for method chaining.

```php
public transformLabels(callable $fn) : self
```

!!! note
    The callback function called for each individual label and should return the transformed label as a continuous or categorical value.

```php
$dataset->transformLabels('intval');

//

$dataset->transformLabels(function ($label) {
    return $label > 0.5 ? 'yes' : 'no';
});
```

### Describe by Label

Describe the features of the dataset broken down by categorical label.

```php
public describeByLabelClasses() : Report
```

```php
echo $dataset->describeByLabelClasses();
```

```json
{
    "not monster": [
        {
            "type": "categorical",
            "num categories": 2,
            "probabilities": {
                "friendly": 0.75,
                "loner": 0.25
            }
        },
        {
            "type": "continuous",
            "mean": 1.125,
            "variance": 12.776875,
            "standard deviation": 3.574475485997911,
            "skewness": -1.0795676577113944,
            "kurtosis": -0.7175867765792474,
            "min": -5,
            "25%": 0.6999999999999993,
            "median": 2.75,
            "75%": 3.175,
            "max": 4
        }
    ],
    "monster": [
        {
            "type": "categorical",
            "num categories": 2,
            "probabilities": {
                "loner": 0.5,
                "friendly": 0.5
            }
        },
        {
            "type": "continuous",
            "mean": -1.25,
            "standard deviation": 0.25,
            "skewness": 0,
            "kurtosis": -2,
            "min": -1.5,
            "25%": -1.375,
            "median": -1.25,
            "75%": -1.125,
            "max": -1
        }
    ]
}
```
