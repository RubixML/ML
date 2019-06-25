<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Labeled.php">Source</a></span>

# Labeled
For *supervised* Estimators you will need to train it with a Labeled dataset consisting of samples with the addition of labels that correspond to the observed outcome of each sample. Splitting, folding, randomizing, sorting, and subsampling are all done while keeping the indices of samples and labels aligned. In addition to the basic Dataset interface, the Labeled class can sort and *stratify* the data by label as well.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | samples | | array | A 2-dimensional array consisting of rows of samples and columns with feature values. |
| 2 | labels | | array | A 1-dimensional array of labels that correspond to the samples in the dataset. |
| 3 | validate | true | bool | Should we validate the data? |

### Additional Methods

#### Factories
Build a new labeled dataset with validation:
```php
public static build(array $samples = [], array $labels = []) : self
```

Build a new labeled dataset foregoing validation:
```php
public static quick(array $samples = [], array $labels = []) : self
```

Build a dataset using a pair of iterators:
```php
public static fromIterator(iterable $samples, iterable $labels) : self
```

**Example**

```php
use Rubix\ML\Datasets\Labeled;

// Import samples and labels

$dataset = new Labeled($samples, $labels, true);  // Using the constructor

$dataset = Labeled::build($samples, $labels);  // Build a new dataset with validation

$dataset = Labeled::quick($samples, $labels);  // Build a new dataset without validation

$dataset = Labeled::fromItertor($samples, $labels); // From a pair of iterators
```

#### Selectors
Return an array of labels:
```php
public labels() : array
```

Zip the samples and labels together in a Generator:
```php
public zip() : Generator
```

Return the label at the given row offset:
```php
public label(int $index) : mixed
```

Return the type of the label encoded as an integer:
```php
public labelType() : int
```

Return all of the possible outcomes i.e. the unique labels:
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

**Output**

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

#### Transform
Transform the labels in the dataset using a callback function:
```php
public transformLabels(callable $fn) : void
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

#### Filter
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

#### Sorting
Sort the dataset by label:
```php
public sortByLabel(bool $descending = false) : self
```

#### Stratification
Group the samples by label and return them in their own dataset:
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
// Put each sample with label 'x' into its own dataset
$strata = $dataset->stratify();

// Fold the dataset into 5 equal size stratified subsets
$folds = $dataset->stratifiedFold(5);

// Split the dataset into two stratified subsets
[$left, $right] = $dataset->stratifiedSplit(0.8);
```