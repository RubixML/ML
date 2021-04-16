# Dataset Objects
Data are passed in specialized in-memory containers called Dataset objects. Dataset objects are table-like data structures that have operations for data manipulation. They can hold a heterogeneous mix of data types and they make it easy to transport data in a canonical way. Datasets consist of a matrix of samples in which each row constitutes a sample and each column represents the value of the feature represented by that column. They have the additional constraint that each feature column must contain values of the same high-level data type. Some datasets can contain labels for training or cross validation. In the example below, we instantiate a new [Labeled](labeled.md) dataset object by passing the samples and their labels as arguments to the constructor.

```php
use Rubix\ML\Datasets\Labeled;

$samples = [
    [0.1, 20, 'furry'],
    [2.0, -5, 'rough'],
];

$labels = ['not monster', 'monster'];

$dataset = new Labeled($samples, $labels);
```

## Factory Methods
Build a dataset with the rows from a 2-dimensional iterable data table:
```php
public static fromIterator(Traversable $iterator) : self
```

!!! note
    When building a [Labeled](labeled.md) dataset, the label values should be in the last column of the data table.

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Extractors\CSV;

$dataset = Labeled::fromIterator(new CSV('example.csv'));
```

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

```php
use Rubix\ML\DataType;

$columns = $dataset->columnsByType(DataType::continuous());
```

## Properties
Return the number of rows in the dataset:
```php
public numSamples() : int
```

Return the number of columns in the samples matrix:
```php
public numFeatures() : int
```

```php
$m = $dataset->numSamples();

$n = $dataset->numFeatures();
```

Return a 2-tuple with the *shape* of the samples matrix:
```php
public shape() : array
```

```php
[$m, $n] = $dataset->shape();

var_dump($m, $n);
```

```sh
int(1000)
int(30)
```

Return the data types for each feature column:
```php
public featureTypes() : array
```

Return the data type for a given column offset:
```php
public featureType(int $offset) : DataType
```

```php
echo $dataset->featureType(15);
```

```sh
categorical
```

## Applying Transformations
You can apply a [Transformer](../transformers/api.md) to the samples in a Dataset object by passing it as an argument to the `apply()` method on the dataset object. If a [Stateful](../transformers/api.md#stateful) transformer has not been fitted beforehand, it will automatically be fitted before being applied to the samples.
```php
public apply(Transformer $transformer) : self
```

```php
use Rubix\ML\Transformers\RobustStandardizer;

$dataset->apply(new RobustStandardizer);
```

## Head and Tail
Return the first *n* rows of data in a new dataset object:
```php
public head(int $n = 10) : self
```

Return the last *n* rows of data in a new dataset object:
```php
public tail(int $n = 10) : self
```

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

## Splitting
Split the dataset into left and right subsets:
```php
public split(float $ratio = 0.5) : array
```

```php
[$training, $testing] = $dataset->split(0.8);
```

## Folding
Fold the dataset to form *k* equal size datasets:
```php
public fold(int $k = 10) : array
```

!!! note
    If there are not enough samples to completely fill the last fold of the dataset then it will contain slightly fewer samples than the rest of the folds.

```php
$folds = $dataset->fold(8);
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

## Batching
Batch the dataset into subsets containing a maximum of *n* rows per batch:
```php
public batch(int $n = 50) : array
```

```php
$batches = $dataset->batch(250);
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

```php
$subset = $dataset->randomSubset(50);
```

Generate a random subset with replacement:
```php
public randomSubsetWithReplacement(int $n) : self
```

```php
$subset = $dataset->randomSubsetWithReplacement(500);
```

Generate a random *weighted* subset with replacement of size *n*:
```php
public randomWeightedSubsetWithReplacement(int $n, array $weights) : self
```

```php
$subset = $dataset->randomWeightedSubsetWithReplacement(200, $weights);
```

## Mapping and Filtering
Map a callback function over the records of the dataset and return the result in a new dataset object:
```php
public map(callable $callback) : self 
```

```php
$addMeanColumn = function ($record) {
    $record[] = array_sum($record) / count($record);

    return $record;
};

$dataset = $dataset->map($addMeanColumn);
```

Filter the records of the dataset using a callback function to determine if a row should be included in the return dataset:
```php
public filter(callable $callback) : self
```

```php
$tallPeople = function ($record) {
	return $record[3] > 178.5;
};

$dataset = $dataset->filter($tallPeople);
```

## Stacking
Stack any number of dataset objects on top of each other to form a single dataset:
```php
public static stack(array $datasets) : self
```

!!! note
    Datasets must have the same number of feature columns i.e. dimensionality.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::stack([
    $dataset1,
    $dataset2,
    $dataset3,
    // ...
]);
```

## Merging and Joining
To merge the rows of this dataset with another dataset:
```php
public merge(Dataset $dataset) : self
```

!!! note
    Datasets must have the same number of columns.

```php
$dataset = $dataset1->merge($dataset2);
```

To join the columns of this dataset with another dataset:
```php
public join(Dataset $dataset) : self
```

!!! note
    Datasets must have the same number of rows.

```php
$dataset = $dataset1->join($dataset2);
```

## Descriptive Statistics
Return an array of statistics such as the central tendency, dispersion and shape of each continuous feature column and the joint probabilities of each category for every categorical feature column:
```php
public describe() : Report
```

```php
echo $dataset->describe();
```

```json
[
    {
        "offset": 0,
        "type": "categorical",
        "num categories": 2,
        "probabilities": {
            "friendly": 0.6666666666666666,
            "loner": 0.3333333333333333
        }
    },
    {
        "offset": 1,
        "type": "continuous",
        "mean": 0.3333333333333333,
        "variance": 9.792222222222222,
        "stddev": 3.129252661934191,
        "skewness": -0.4481030843690633,
        "kurtosis": -1.1330702741786107,
        "min": -5,
        "25%": -1.375,
        "median": 0.8,
        "75%": 2.825,
        "max": 4
    }
]
```

## Sorting
To sort a dataset in place by a specific feature column:
```php
public sortByColumn(int $offset, bool $descending = false) : self
```

```php
$dataset->sortByColumn(5, true);
```

## De-duplication
Remove duplicate rows from the dataset:
```php
public deduplicate() : self
```

## Saving
Export the dataset to the location and format given by a [Writable](../extractors/api.md) extractor:
```php
public exportTo(Writable $extractor) : void
```

```php
use Rubix\ML\Extractors\NDJSON;

$dataset->exportTo(new NDJSON('example.ndjson'));
```
