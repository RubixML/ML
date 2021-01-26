# Extractors
Extractors are data table iterators that help you import data from various source formats such as CSV, JSON, and NDJSON in an efficient way. They implement one of the standard PHP [Traversable](https://www.php.net/manual/en/class.traversable.php) interfaces and can be used to instantiate a new [Dataset](../datasets/api.md) object by passing it to the `fromIterator()` method.

!!! note
    Extractors are read-only, they will never overwrite the source dataset.

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;

$dataset = Labeled::fromIterator(new NDJSON('example.ndjson'));
```

### Iterate
Extractors can also be used on their own to loop through the records of a data table. In the example below, we show how to iterate over the records of a CSV file.

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV('example.csv');

foreach ($extractor as $record) {
    // ...
}
```

### Return an Iterator
To return the underlying iterator wrapped by the extractor object:
```php
public getIterator() : Traversable
```

The example below shows how you can instantiate a new dataset object using only a portion of the source dataset by wrapping an underlying iterator with the standard PHP library's [Limit Iterator](https://www.php.net/manual/en/class.limititerator.php).

```php
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Datasets\Unlabeled;
use LimitIterator;

$extractor = new NDJSON('example.ndjson');

$iterator = new LimitIterator($extractor->getIterator(), 500, 1000);

$dataset = Unlabeled::fromIterator($iterator);
```