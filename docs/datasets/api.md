# Dataset Objects
Data are passed in specialized in-memory containers called Dataset objects. Dataset objects are table-like structures employing a high-level type system that have operations for data manipulation. They can hold a heterogeneous mix of data types and they make it easy to transport data in a canonical way. Datasets require a table of samples in which each row constitutes a sample and each column represents the value of the feature represented by that column. They have the additional constraint that each feature column must be homogenous i.e. they must contain values of the same high-level data type. For example, a continuous feature column must only contain integer or floating point numbers. A stray string or other data type will throw an exception upon validation.

In the example below, we instantiate a new [Labeled](labeled.md) dataset object by passing the samples and their labels to the constructor.

```php
use Rubix\ML\Datasets\Labeled;

$samples = [
    [0.1, 20, 'furry'],
    [2.0, -5, 'rough'],
];

$labels = ['not monster', 'monster'];

$dataset = new Labeled($samples, $labels);
```

## Missing Values
By convention, missing continuous values are denoted by `NaN` and missing categorical values are denoted by a special placeholder category (ex. the `?` category). Dataset objects do not allow missing values of resource or other data types.

```php
$samples = [
    [0.001, NAN, 'rough'], // Missing a continuous value
    [0.25, -1000, '?'], // Missing a categorical value
    [0.01, -500, 'furry'], // Complete sample
];
```

## Factory Methods
Build a dataset with the rows from a 2-dimensional iterable data table:
```php
public static fromIterator(Traversable $iterator) : self
```

**Example**

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Extractors\CSV;

$dataset = Labeled::fromIterator(new CSV('example.csv'));
```

**Note:** The data must be in the format of a table where each row is an n-d array of values. By convention, labels are always the last column of the data table.

## Selecting
Return all the samples in the dataset in a 2-dimensional array:
```php
public samples() : array
```

Select a single row containing the sample at a given offset (offsets begin at 0):
```php
public sample(int $offset) : array
```

Select the values of a feature column at a given offset (offsets begin at 0):
```php
public column(int $offset) : array
```

Return the columns of the sample matrix:
```php
public columns() : array
```

Return the columns of the sample matrix of a particular type:
```php
public columnsByType(DataType $type) : array
```

**Example**

```php
use Rubix\ML\DataType;

$columns = $dataset->columnsByType(DataType::continuous());
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

Return the shape of the dataset:
```php
public shape() : array
```

Return the data types for each feature column:
```php
public columnTypes() : array
```

Return the data type for a given column offset:
```php
public columnType(int $offset) : DataType
```

**Example**

```php
$m = $dataset->numRows();

$n = $dataset->numColumns();

[$m, $n] = $dataset->shape();

var_dump($m, $n);

echo $dataset->columnType(15);
```

```sh
int(1000)
int(30)

categorical
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

**Examples**

```php
$dataset = $dataset->transformColumn(0, 'log1p');

$dataset = $dataset->transformColumn(6, function ($value) {
    return $value === 0 ? NAN : $value;
});

$dataset = $dataset->transformColumn(5, function ($value) {
    return min($value, 1000);
});
```

## Stacking Datasets
Stack any number of dataset objects on top of each other to form a single dataset:
```php
public static stack(array $datasets) : self
```

> **Note:** Datasets must have the same number of feature columns i.e. dimensionality.

**Example**

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::stack([
    $dataset1,
    $dataset2,
    $dataset3,
    // ...
]);
```

## Merging Datasets
To merge the rows of this dataset with another dataset:
```php
public merge(Dataset $dataset) : self
```

> **Note:** Datasets must have same number of columns to merge.

To merge the columns of this dataset with another dataset:
```php
public augment(Dataset $dataset) : self
```

> **Note:** Datasets must have same number of rows to augment.

**Examples**

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;

$dataset = $dataset->merge(new Labeled($samples,  $labels));

$dataset = $dataset->augment(new Unlabeled($samples));
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

**Examples**

```php
$subset = $dataset->head(10);

$subset = $dataset->tail(30);
```

## Taking and Leaving
Remove *n* rows from the dataset and return them in a new dataset:
```php
public take(int $n = 1) : self
```

Leave *n* samples on the dataset and return the rest in a new dataset:
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
Split the dataset into left and right subsets:
```php
public split(float $ratio = 0.5) : array
```

Partition the dataset into left and right subsets using the values of a single feature column for comparison:
```php
public partitionByColumn(int $offset, mixed $value) : array
```

Partition the dataset into left and right subsets based on the samples' distances from two centroids:
```php
public spatialPartition(array $leftCentroid, array $rightCentroid, ?Distance $kernel = null) : array
```

**Examples**

```php
use Rubix\ML\Kernels\Distance\Euclidean;

[$training, $testing] = $dataset->split(0.8);

[$left, $right] = $dataset->partitionByColumn(4, 50);

[$left, $right] = $dataset->spatialPartition($leftCentroid, $rightCentroid, new Euclidean());
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

foreach ($folds as $fold) {
    // ...
}
```

## Batching
Batch the dataset into subsets containing a maximum of *n* rows per batch:
```php
public batch(int $n = 50) : array
```

**Example**

```php
$batches = $dataset->batch(250);

foreach ($batches as $batch) {
    // ...
}
```

## Randomization
Randomize the order of the dataset and return it for method chaining:
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
$dataset = $dataset->randomize();

$subset = $dataset->randomSubset(50);

$subset = $dataset->randomSubsetWithReplacement(500);

$subset = $dataset->randomWeightedSubsetWithReplacement(200, $weights);
```

## Filtering
Filter the rows of the dataset using the values of a feature column at the given offset as the arguments to a filter callback. The callback should return false for rows that should be filtered.
```php
public filterByColumn(int $offset, callable $fn) : self
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
public sortByColumn(int $offset, bool $descending = false) : self
```

**Example**

```php
var_dump($dataset->samples());

$dataset->sortByColumn(2);

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
Drop the row at the given offset:
```php
public dropRow(int $offset) : self
```

Drop the rows at the given offsets:
```php
public dropRows(array $indices) : self
```

Drop the column at the given offset:
```php
public dropColumn(int $offset) : self
```

Drop the columns at the given indices:
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
print_r($dataset->describe());
```

```sh
Array
(
    [2] => Array
        (
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
Remove duplicate rows from the dataset:
```php
public deduplicate() : self
```

## Output Formats
Return the dataset object as a data table array:
```php
public toArray() : array
```

Return a JSON representation of the dataset:
```php
public toJSON(bool $pretty = false) : string
```

Return a newline delimited JSON representation of the dataset:
```php
public toNDJSON() : string
```

Return the dataset as comma-separated values (CSV) string:
```php
public toCSV(string $delimiter = ',', string $enclosure = '"') : string
```

**Example**

```php
file_put_contents('dataset.ndjson', $dataset->toNDJSON());
```

## Previewing in the Console
You can echo the dataset object to preview the first few rows and columns in the console.

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
