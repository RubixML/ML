<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Datasets/Labeled.php">[source]</a></span>

# Labeled
A Labeled dataset is used to train supervised learners and for testing a model by providing the ground-truth. In addition to the standard dataset API, a labeled dataset can perform operations such as stratification and sorting the dataset using the label column.

!!! note
    Since PHP silently converts integer strings (ex. `'1'`) to integers in some circumstances, you should not use integer strings as class labels. Instead, use an appropriate non-integer string class name such as `'class 1'`, `'#1'`, or `'first'`.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
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
Return the labels of the dataset in an array:
```php
public labels() : array
```

Return a single label at the given row offset:
```php
public label(int $offset) : mixed
```

Return the data type of the label:
```php
public labelType() : DataType
```

```php
echo $dataset->labelType();
```

```sh
continuous
```

Return all of the possible outcomes i.e. the unique labels in an array:
```php
public possibleOutcomes() : array
```

```php
var_dump($dataset->possibleOutcomes());
```

```sh
array(2) {
	[0]=> string(5) "female"
	[1]=> string(4) "male"
}
```

### Stratification
Group samples by their class label and return them in their own dataset:
```php
public stratify() : array
```

```php
$strata = $dataset->stratify();
```

Split the dataset into left and right subsets such that the proportions of class labels remain intact:
```php
public stratifiedSplit($ratio = 0.5) : array
```

```php
[$training, $testing] = $dataset->stratifiedSplit(0.8);
```

Return *k* equal size subsets of the dataset such that class proportions remain intact:
```php
public stratifiedFold($k = 10) : array
```

```php
$folds = $dataset->stratifiedFold(3);
```

### Transform Labels
Transform the labels in the dataset using a callback function and return self for method chaining:
```php
public transformLabels(callable $fn) : self
```

!!! note
    The callback function called for each individual label and should return the transformed label as a continuous or categorical value.

```php
$dataset->transformLabels('intval');

$dataset->transformLabels('floatval');

$dataset->transformLabels(function ($label) {
	switch ($label) {
		case 0:
			return 'disagree';

		case 1:
            return 'neutral';
            
        case 2:
            return 'agree';
            
        default:
            return '?';
	}
});

$dataset->transformLabels(function ($label) {
	return $label > 0.5 ? 'yes' : 'no';
});
```

### Filter
Filter the dataset by label:
```php
public filterByLabel(callable $fn) : self
```

!!! note
    The callback function is given a label as its only argument and should return true if the row should be kept or false if the row should be filtered out of the result.

```php
$filtered = $dataset->filterByLabel(function ($label)) {
	return $label <= 10000;;
});
```

### Sorting
Sort the dataset by label and return self for method chaining:
```php
public sortByLabel(bool $descending = false) : self
```

### Describe by Label
Describe the features of the dataset broken down by label:
```php
public describeByLabel() : Report
```

```php
echo $dataset->describeByLabel();
```

```json
{
    "not monster": [
        {
            "type": "categorical",
            "num_categories": 2,
            "probabilities": {
                "friendly": 0.75,
                "loner": 0.25
            }
        },
        {
            "type": "continuous",
            "mean": 1.125,
            "variance": 12.776875,
            "std_dev": 3.574475485997911,
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
            "num_categories": 2,
            "probabilities": {
                "loner": 0.5,
                "friendly": 0.5
            }
        },
        {
            "type": "continuous",
            "mean": -1.25,
            "variance": 0.0625,
            "std_dev": 0.25,
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

### Describe the Labels
Return an array of descriptive statistics about the labels in the dataset:
```php
public describeLabels() : Report
```

```php
echo $dataset->describeLabels();
```

```json
{
    "type": "categorical",
    "num_categories": 2,
    "probabilities": {
        "not monster": 0.6666666666666666,
        "monster": 0.3333333333333333
    }
}
```