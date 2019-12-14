# Dataset Objects
In Rubix ML, data are passed in specialized in-memory containers called Dataset objects. Dataset objects are extended table-like data structures employing an internal type system and many operations for data manipulation. They can hold a heterogeneous mix of data types and they make it easy to transport data in a canonical way. Each dataset requires a table of samples where each row constitues a single sample and each column represents the value of the feature indexed at that column.

Dataset objects have the additional constraint that each feature column must be homogenous i.e. they must contain values of the same data type. For example, a continuous feature column's value for height must be either an integer or floating point number. A stray string or other data type will throw an exception upon validation.

**Example**

```php
use Rubix\ML\Datasets\Unlabeled;

$samples = [
    [0.1, 20, 'furry'],
    [2.0, -5, 'rough'],
    // ...
];

$dataset = new Unlabeled($samples);
```

## Missing Values
By convention, continuous missing values are denoted by `NaN` and categorical values are denoted by a special placeholder category (ex. the `?` category). Dataset objects do not allow missing values of resource or other data types.

**Example**

```php
$samples = [
    [0.001, NAN, 'rough'], // Missing a continuous value
    [0.25, -1000, '?'], // Missing a categorical value
    [0.01, -500, 'furry'], // Complete sample
];
```

## Factory Methods
Build a dataset object from a JSON string:
```php
public static fromJson(string $json) : self
```

**Example**

```php
use Rubix\ML\Datasets\Labeled;

$json = '{
    "samples": [
        [
            "nice",
            "furry",
            "friendly",
            4
        ],
        [
            "mean",
            "furry",
            "loner",
            -1.5
        ]
    ],
    "labels": [
        "not monster",
        "monster"
    ]
}';

$dataset = Labeled::fromJson($json);
```

Build a dataset from a newline delimited JSON string:
```php
public static fromNdjson(string $ndjson) : self
```

**Example**

```php
use Rubix\ML\Datasets\Unlabeled;

$ndjson = '["nice","furry","friendly",4]\n'
    . '["mean","furry","loner",-1.5]\n'
    . '["nice","rough","friendly",2.6]\n'
    . '["mean","rough","friendly",-1]\n'
    . '["nice","rough","friendly",2.9]\n'
    . '["nice","furry","loner",-5]\n';

$dataset = Unlabeled::fromNdjson($ndjson);
```

Build a dataset object from a CSV (comma-separated values) string:
```php
public static fromCsv(string $csv, string $delimiter = ',', string $enclosure = '') : self
```

**Example**

```php
use Rubix\ML\Datasets\Labeled;

$csv = 'nice,furry,friendly,4,not monster\n'
    . 'mean,furry,loner,-1.5,monster\n'
    . 'nice,rough,friendly,2.6,not monster\n';

$dataset = Labeled::fromCsv($csv);
```

**Note:** In a [Labeled](labeled.md) dataset the *last* column of a CSV table are taken as the labels.

## Selecting
Return all the samples in the dataset in a 2-dimensional array:
```php
public samples() : array
```

Select a single row containing the sample at a given offset (offsets begin at 0):
```php
public sample(int $index) : array
```

Select the values of a feature column at a given offset (offsets begin at 0):
```php
public column(int $index) : array
```

## Properties
Return the number of rows in the dataset:
```php
public numRows() : int
```

Return the number of columns in the dataset:
```php
public numColumns() : int
```

Return the integer encoded column types for each feature column:
```php
public types() : array
```

Return the integer encoded column type given a column index:
```php
public columnType(int $index) : int
```

## Applying Transformations
You can apply a [Transformer](#transformers) directly to a Dataset by passing it to the `apply()` method on the dataset object. The method returns self for chaining.

```php
public apply(Transformer $transformer) : self
```

**Example**

```php
use Rubix\ML\Transformers\RandomHotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new RandomHotDeckImputer())
    ->apply(new OneHotEncoder());
```

You can also transform a single feature column using a callback function with the `transformColumn()` method.

```php
public transformColumn(int $column, callable $callback) : self
```

**Example**

```php
$dataset = $dataset->transformColumn(0, 'log1p'); // Log transform column 0

$dataset = $dataset->transformColumn(6, function ($value) {
    return $value === 0 ? NAN : $value; // Replace 0 with NaN
});

$dataset = $dataset->transformColumn(15, function ($value) {
    return $value === 'NA' ? '?' : $value; // Replace the 'NA' category with '?'
});

$dataset = $dataset->transformColumn(32, function ($value) {
    return min($value, 1000); // Cap values at 1000
});
```

## Stacking
Stack any number of dataset objects on top of each other to form a single dataset:
```php
public static stack(array $datasets) : self
```

> **Note:** Datasets must have the same number of feature columns i.e. dimensionality.

**Example**

```php
use Rubix\ML\Datasets\Labeled;

// Import training datasets

$dataset = Labeled::stack([
    $training1,
    $training2,
    $training3,
]);
```

## Prepending and Appending
To prepend a given dataset onto the beginning of another dataset:
```php
public prepend(Dataset $dataset) : self
```

To append a given dataset onto the end of another dataset:
```php
public append(Dataset $dataset) : self
```

> **Note:** Datasets must have the same number of feature columns i.e. dimensionality.

**Example**

```php
$dataset = $training->append($testing); // Append the testing set to the training set

$dataset = $dataset->prepend($new); // Prepend a new dataset to the current dataset
```

## Head and Tail
Return the *first* **n** rows of data in a new dataset object:
```php
public head(int $n = 10) : self
```

Return the *last* **n** rows of data in a new dataset object:
```php
public tail(int $n = 10) : self
```

**Example**

```php
// Return the first 5 rows in a new dataset
$subset = $dataset->head(5);

// Return the last 10 rows in a new dataset
$subset = $dataset->tail(10);
```

## Taking and Leaving
Remove **n** rows from the dataset and return them in a new dataset:
```php
public take(int $n = 1) : self
```

Leave **n** samples on the dataset and return the rest in a new dataset:
```php
public leave(int $n = 1) : self
```

## Slicing and Splicing
Return an *n* size portion of the dataset in a new dataset:
```php
public slice(int $offset, int $n) : self
```

Remove a size *n* chunk of the dataset starting at *offset* and return it in a new dataset:
```php
public splice(int $offset, int $n) : self
```

## Splitting
Split the dataset into left and right subsets given by a *ratio*:
```php
public split(float $ratio = 0.5) : array
```

Partition the dataset into left and right subsets based on the value of a feature in a specified column:
```php
public partition(int $index, mixed $value) : array
```

**Example**

```php
// Split the dataset 50/50 into left and right subsets
[$left, $right] = $dataset->split(0.5);

// Split the dataset into training and testing sets 80/20.
[$training, $testing] = $dataset->split(0.8);

// Partition the dataset by the feature column at index 4 by value '50'
[$left, $right] = $dataset->partition(4, 50);
```

## Folding
Fold the dataset to form *k* equal size datasets:
```php
public fold(int $k = 10) : array
```

> **Note:** If there are not enough samples to completely fill the last fold of the dataset then it will contain slightly fewer samples than the rest.

**Example**

```php
$folds = $dataset->fold(8);

var_dump(count($folds));
```

```sh
int(8)
```

## Batching
Batch the dataset into subsets containing a maximum of *n* rows per batch:
```php
public batch(int $n = 50) : array
```

**Example**

```php
$batches = $dataset->batch(1000);
```

## Randomization
Randomize the order of the Dataset and return it for method chaining:
```php
public randomize() : self
```

Generate a random subset of the dataset without replacement of size *n*:
```php
public randomSubset(int $n) : self
```

Generate a random subset with replacement:
```php
public randomSubsetWithReplacement($n) : self
```

Generate a random *weighted* subset with replacement of size *n*:
```php
public randomWeightedSubsetWithReplacement($n, array $weights) : self
```

**Example**

```php
// Randomize and split the dataset into two subsets
[$left, $right] = $dataset->randomize()->split(0.6);

// Generate a random unique subset of 50 random samples
$subset = $dataset->randomSubset(50);

// Generate a 'bootstrap' dataset of 500 random samples
$subset = $dataset->randomSubsetWithReplacement(500);

// Sample a random subset according to a user-defined weight distribution
$subset = $dataset->randomWeightedSubsetWithReplacement(200, $weights);

// Sample a random subset using the values of a column as sample weights
$subset = $dataset->randomWeightedSubsetWithReplacement(200, $dataset->column(1));
```

## Filtering
To filter a Dataset by a feature column:
```php
public filterByColumn(int $index, callable $fn) : self
```

**Example**

```php
$tallPeople = $dataset->filterByColumn(3, function ($value) {
	return $value > 178.5;
});
```

## Sorting
To sort a dataset in place by a specific feature column:
```php
public sortByColumn(int $index, bool $descending = false) : self
```

**Example**

```php
var_dump($dataset->samples());

$dataset->sortByColumn(2, false);

var_dump($dataset->samples());
```

```sh
array(3) {
    [0]=> array(3) {
	    [0]=> string(4) "mean"
	    [1]=> string(4) "furry"
	    [2]=> int(8)
    }
    [1]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(1)
    }
    [2]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(6)
    }
}

array(3) {
    [0]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(1)
    }
    [1]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(6)
    }
    [2]=> array(3) {
	    [0]=> string(4) "mean"
	    [1]=> string(4) "furry"
	    [2]=> int(8)
    }
}
```

## Dropping Rows and Columns
Drop the row at the given index and return the new dataset:
```php
public dropRow(int $index) : self
```

Drop the rows at the given indices and return the new dataset:
```php
public dropRows(array $indices) : self
```

Drop the column at the given index and return the new dataset:
```php
public dropColumn(int $index) : self
```

Drop the columns at the given indices and return the new dataset:
```php
public dropColumns(array $indices) : self
```

## Descriptive Statistics
Return an array of statistics such as the central tendency, dispersion and shape of each continuous feature column and the joint probabilities of each category for every categorical feature column:
```php
public describe() : array
```

**Example**

```php
$stats = $dataset->describe();

print_r($stats);
```

```sh
Array
(
	...

    [2] => Array
        (
            [column] => 2
            [type] => categorical
            [num_categories] => 2
            [probabilities] => Array
                (
                    [friendly] => 0.66666666666667
                    [loner] => 0.33333333333333
                )

        )

    [3] => Array
        (
            [column] => 3
            [type] => continuous
            [mean] => 0.33333333333333
            [variance] => 9.7922222222222
            [std_dev] => 3.1292526619342
            [skewness] => -0.44810308436906
            [kurtosis] => -1.1330702741786
            [min] => -5
            [25%] => -1.375
            [median] => 0.8
            [75%] => 2.825
            [max] => 4
        )

)
```

## De-duplication
Return a dataset with duplicate rows removed:
```php
public deduplicate() : self
```

## Output Formats
Return the dataset object as an associative array:
```php
public toArray() : array
```

Return a JSON representation of the dataset:
```php
public toJson(bool $pretty = false) : string
```

Return a newline delimited JSON representation of the dataset:
```php
public toNdjson() : string
```

Return the dataset as comma-separated values (CSV) string:
```php
public toCsv(string $delimiter = ',', string $enclosure = '') : string
```

## Previewing in the Console
You can echo the dataset objectto preview the first rows and columns in the console.

```php
echo $dataset;
```

```sh
| Column 0    | Column 1    | Column 2    | Column 3    | Label       |
-----------------------------------------------------------------------
| nice        | furry       | friendly    | 4           | not monster |
-----------------------------------------------------------------------
| mean        | furry       | loner       | -1.5        | monster     |
-----------------------------------------------------------------------
| nice        | rough       | friendly    | 2.6         | not monster |
-----------------------------------------------------------------------
| mean        | rough       | friendly    | -1          | monster     |
-----------------------------------------------------------------------
| nice        | rough       | friendly    | 2.9         | not monster |
-----------------------------------------------------------------------
| nice        | furry       | loner       | -5          | not monster |
-----------------------------------------------------------------------
```
