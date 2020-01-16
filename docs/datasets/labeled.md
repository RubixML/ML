<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Labeled.php">[source]</a></span>

# Labeled
A Labeled dataset is used to train supervised learners and for testing a model by providing the ground-truth. In addition to the standard dataset API, a Labeled dataset can perform operations such as stratification and sorting the dataset by the label column.

> **Note:** Labels can be either categorical or continuous data types but NaN values are not allowed.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | samples | | array | A 2-dimensional array consisting of rows of samples and columns with feature values. |
| 2 | labels | | array | A 1-dimensional array of labels that correspond to each sample in the dataset. |
| 3 | validate | true | bool | Should we validate the data? |

## Additional Methods

### Factory Methods
Build a new labeled dataset with validation:
```php
public static build(array $samples = [], array $labels = []) : self
```

Build a new labeled dataset foregoing validation:
```php
public static quick(array $samples = [], array $labels = []) : self
```

**Example**

```php
use Rubix\ML\Datasets\Labeled;

$samples = [
    [0.1, 20, 'furry'],
    [2.0, -5, 'rough'],
    [0.01, 5, 'furry'],
];

$labels = ['not monster', 'monster', 'not monster'];


$dataset = new Labeled($samples, $labels); // With validation

$dataset = new Labeled($samples, $labels, false); // Without validation

$dataset = Labeled::build($samples, $labels);  // With validation

$dataset = Labeled::quick($samples, $labels);  // Without validation
```

### Selectors
Return an array of labels:
```php
public labels() : array
```

Return a single label at the given row offset:
```php
public label(int $index) : mixed
```

Return the data type of the label:
```php
public labelType() : DataType
```

Return all of the possible outcomes i.e. the unique labels in an array:
```php
public possibleOutcomes() : array
```

**Example**

```php
// Return the labels in the dataset
$labels = $dataset->labels();

// Return the label at row offset 3
$label = $dataset->label(3);

// Return an array of unique labels
$outcomes = $dataset->possibleOutcomes();

var_dump($labels);
var_dump($label);
var_dump($outcomes);
```

```sh
array(4) {
    [0]=> string(5) "female"
    [1]=> string(4) "male"
    [2]=> string(5) "female"
    [3]=> string(4) "male"
}

string(4) "male"

array(2) {
	[0]=> string(5) "female"
	[1]=> string(4) "male"
}
```

### Transform
Transform the labels in the dataset using a callback function and return self for method chaining:
```php
public transformLabels(callable $fn) : self
```

> **Note:** The callback function is given a label as its only argument and should return the transformed label as a continuous or categorical value.

**Example**

```php
$dataset->transformLabels('intval'); // To integers

$dataset->transformLabels('floatval'); // To floats

// From integers to discrete classes
$dataset->transformLabels(function ($label) {
	switch ($label) {
		case 1:
			return 'male';

		case 2:
			return 'female';

		default:
			return 'other';
	}
});

// From a continuous value to binary classes
$dataset->transformLabels(function ($label) {
	return $label > 0.5 ? 'yes' : 'no';
});
```

### Filter
Filter the dataset by label:
```php
public filterByLabel(callable $fn) : self
```

> **Note:** The callback function is given a label as its only argument and should return true if the row should be kept or false if the row should be filtered out of the result.

**Example**

```php
// Remove rows with label values greater than 10000
$filtered = $dataset->filterByLabel(function ($label)) {
	return $label > 10000 ? false : true;
});
```

### Sorting
Sort the dataset by label and return self for method chaining:
```php
public sortByLabel(bool $descending = false) : self
```

### Stratification
Group samples by their label and return them in their own datasets:
```php
public stratify() : array
```

Split the dataset into left and right stratified subsets with a given *ratio* of samples in each:
```php
public stratifiedSplit($ratio = 0.5) : array
```

Return *k* equal size subsets of the dataset:
```php
public stratifiedFold($k = 10) : array
```

**Example**

```php
// Put each sample with a given label into its own dataset
$strata = $dataset->stratify();

// Fold the dataset into 5 equal-sized stratified subsets
$folds = $dataset->stratifiedFold(5);

// Split the dataset into two 50/50 stratified subsets
[$left, $right] = $dataset->stratifiedSplit(0.5);

// Split the dataset into two stratified training and testing sets
[$training, $testing] = $dataset->stratifiedSplit(0.8);
```

### Describe by Label
Describe the features of the dataset broken down by label:
Return an array of descriptive statistics about the labels in the dataset:
```php
public describeByLabel() : array
```

**Example**

```php
print_r($dataset->describeByLabel());
```

```sh
Array
(
    [not monster] => Array
        (
            [0] => Array
                (
                    [type] => categorical
                    [num_categories] => 1
                    [densities] => Array
                        (
                            [nice] => 1
                        )

                )

            [1] => Array
                (
                    [type] => categorical
                    [num_categories] => 2
                    [densities] => Array
                        (
                            [furry] => 0.5
                            [rough] => 0.5
                        )

                )

            [2] => Array
                (
                    [type] => categorical
                    [num_categories] => 2
                    [densities] => Array
                        (
                            [friendly] => 0.75
                            [loner] => 0.25
                        )

                )

            [3] => Array
                (
                    [type] => continuous
                    [mean] => 1.125
                    [variance] => 12.776875
                    [std_dev] => 3.5744754859979
                    [skewness] => -1.0795676577114
                    [kurtosis] => -0.71758677657925
                    [min] => -5
                    [25%] => 0.7
                    [median] => 2.75
                    [75%] => 3.175
                    [max] => 4
                )

        )

    [monster] => Array
        (
            ...
        )
)
```

### Describe the Labels
Return an array of descriptive statistics about the labels in the dataset:
```php
public describeLabels() : array
```

**Example**

```php
print_r($dataset->describeLabels());
```

```sh
Array
(
    [type] => categorical
    [num_categories] => 2
    [densities] => Array
        (
            [monster] => 0.33333333333333
            [not monster] => 0.66666666666667
        )

)
```