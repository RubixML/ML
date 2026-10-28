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

## Modifying a Dataset

A return type of `self` does not by itself tell you whether a method modifies the dataset. Some methods return the same object for chaining, while others create a new dataset.

| Methods | Effect on the original dataset | Return value |
| --- | --- | --- |
| `apply()`, `reverseApply()`, `dropFeature()`, `randomize()` | Modify the original dataset in place | The same dataset object |
| `filter()`, `sort()`, `deduplicate()`, `head()`, `tail()`, `slice()`, `merge()`, `join()`, `randomSubset()`, `randomSubsetWithReplacement()`, `randomWeightedSubsetWithReplacement()` | Leave the original dataset unchanged | A new dataset object |
| `split()`, `fold()`, `batch()` | Leave the original dataset unchanged | An array of new dataset objects |
| `take()`, `leave()`, `splice()` | Remove rows from the original dataset | A new dataset containing the removed rows |

For labeled datasets, row operations keep each sample paired with its label. The additional `transformLabels()` method modifies the labels in place and returns the same dataset object.

Assign the result of a method such as `filter()` to retain the new dataset. Assigning the result of an in-place method to another variable does not make a copy: both variables refer to the same dataset object.

## Factory Methods

Build a dataset with the records of a 2-dimensional iterable data table:

```php
public static fromIterator(Traversable $iterator, bool $verify = true) : self
```

!!! note
    When building a [Labeled](labeled.md) dataset, the label values should be in the last column of the data table.

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Extractors\CSV;

$dataset = Labeled::fromIterator(new CSV('example.csv'));
```

Stream Dataset chunks that are lazy-loaded from an iterator such as an [Extractor](../extractors/api.md). Useful for training an [Online](../online.md) learner on a dataset that is too large to fit into memory all at once. Because the batches are generated lazily, only n records are ever held in memory at a time. Each batch is validated on construction by default, but you can disable verification for speed if you trust the data by setting `verify` to false.

```php
public static chunked(iterable $iterator, int $size, bool $verify = true) : Generator
```

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;

foreach (Labeled::chunked(new NDJSON('too-large.jsonl'), size: 1024) as $batch) {
    $estimator->partial($batch);
}
```

!!! note
    When building a [Labeled](labeled.md) dataset, the label values should be in the last column of the data table.

## Properties

Return the number of rows in the dataset:

```php
public numSamples() : int
```

Return the number of columns in the samples matrix:

```php
public numFeatures() : int
```

Return a 2-tuple with the *shape* of the samples matrix:

```php
public shape() : array{int, int}
```

```php
[$m, $n] = $dataset->shape();

echo "$m x $n";
```

```text
1000 x 30
```

## Data Types

Return the data types for each column in the data table:

```php
public types() : Rubix\ML\DataType[]
```

Return the data types for each feature column:

```php
public featureTypes() : Rubix\ML\DataType[]
```

Return the data type for a given column offset:

```php
public featureType(int $offset) : Rubix\ML\DataType
```

```php
echo $dataset->featureType(15);
```

```sh
categorical
```

## Selecting

Return all the samples in the dataset in a 2-dimensional array:

```php
public samples() : array[]
```

Select a single row containing the sample at a given offset beginning at 0:

```php
public sample(int $offset) : mixed[]
```

Return the columns of the sample matrix:

```php
public features() : array[]
```

Select the values of a feature column at a given offset :

```php
public feature(int $offset) : mixed[]
```

## Dropping

Drop a feature at a given column offset in place and return the same dataset object:

```php
public dropFeature(int $offset) : self
```

## Head and Tail

Return the first *n* rows of data in a new dataset object:

```php
public head(int $n = 10) : self
```

```php
$subset = $dataset->head(10);
```

Return the last *n* rows of data in a new dataset object:

```php
public tail(int $n = 10) : self
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
public split(float $ratio = 0.5) : array{self, self}
```

```php
[$training, $testing] = $dataset->split(0.8);
```

## Folding

Fold the dataset to form *k* equal size datasets:

```php
public fold(int $k = 10) : self[]
```

!!! note
    Folds are always as equal size as possible. Any samples that do not divide evenly are distributed one per fold starting from the first fold, so no two folds ever differ in size by more than a single sample. If *k* is greater than the number of samples then an `InvalidArgumentException` is thrown.

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
public batch(int $n = 50) : self[]
```

```php
$batches = $dataset->batch(250);
```

## Randomization

Randomize the order of the dataset in place and return the same dataset object for method chaining:

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

Each weight must be non-negative and the sum of the weights must be greater than zero. If *n* is less than 1, there are no samples in the dataset, the number of weights does not equal the number of samples, any weight is negative, or the sum of the weights is not greater than zero then an `InvalidArgumentException` is thrown.

## Applying Transformations

You can apply a [Transformer](../transformers/api.md) to the samples in a Dataset object by passing it as an argument to the `apply()` method on the dataset object. This modifies the samples in place and returns the same dataset object. If a [Stateful](../transformers/api.md#stateful) transformer has not been fitted beforehand, it will automatically be fitted before being applied to the samples.

```php
public apply(Transformer $transformer) : self
```

```php
use Rubix\ML\Transformers\RobustStandardizer;

$dataset->apply(new RobustStandardizer);
```

To reverse the transformation in place, pass a [Reversible](../transformers/api.md#reversible) transformer to the dataset object's `reverseApply()` method. This also returns the same dataset object.

```php
public reverseApply(Reversible $transformer) : self
```

```php
use Rubix\ML\Transformers\MaxAbsoluteScaler;

$transformer = new MaxAbsoluteScaler();

$dataset->apply($transformer);

// Do something

$dataset->reverseApply($transformer);
```

## Filtering

Return a new dataset containing the records accepted by a callback function. The original dataset is unchanged:

```php
public filter(callable $callback) : self
```

```php
$tallPeople = function ($record) {
    return $record[3] > 178.5;
};

$tallDataset = $dataset->filter($tallPeople);
```

Use `$dataset = $dataset->filter($tallPeople)` if you want to replace the variable with the filtered dataset. Other variables referring to the original object still refer to the unfiltered dataset.

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

Return a new dataset containing the rows of this dataset followed by those of another dataset. Both input datasets are unchanged:

```php
public merge(Dataset $dataset) : self
```

!!! note
    Datasets must have the same number of columns.

```php
$dataset = $dataset1->merge($dataset2);
```

Return a new dataset by joining the columns of this dataset with another dataset. Both input datasets are unchanged:

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
public describe() : Rubix\ML\Report
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
        "standard deviation": 3.129252661934191,
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

Return a new dataset with its records sorted using a callback for comparisons between samples. The original dataset is unchanged. The callback function accepts two records to be compared and should return `true` if the records should be swapped.

```php
public function sort(callable $callback) : self
```

```php
$sorted = $dataset->sort(function ($recordA, $recordB) {
    return $recordA[2] > $recordB[2];
});
```

## De-duplication

Return a new dataset with duplicate rows removed. The original dataset is unchanged:

```php
public deduplicate() : self
```

## Exporting

Export the dataset to the location and format given by a [Exporter](../extractors/api.md) extractor. If `overwrite` is true then the current samples will be overwritten, otherwise they will be appended if the file or database already exists.

```php
public exportTo(Exporter $extractor, bool $overwrite = false) : void
```

```php
use Rubix\ML\Extractors\NDJSON;

$dataset->exportTo(new NDJSON('example.jsonl'), false);
```
