# Extractors
Extractors are data table iterators that help you import data from various source formats such as CSV, NDJSON, and SQL in an efficient way. They implement one of the standard PHP [Traversable](https://www.php.net/manual/en/class.traversable.php) interfaces and are compatible anywhere the iterable pseudotype is accepted. Extractors that implement the Writable interface can be used to save other iterators such as dataset objects and other extractors.

## Iterate
Calling `foreach` on an extractor object iterates over the rows of the data table. In the example below, we'll use the [CSV](csv.md) extractor to print out the rows of the dataset to the console.

```php
use Rubix\ML\Extractors\CSV;

foreach (new CSV('example.csv') as $row) {
    print_r($row);
}
```

We can also instantiate a new [Dataset](../datasets/api.md) object by passing an extractor to the `fromIterator()` method.

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;

$dataset = Labeled::fromIterator(new NDJSON('example.ndjson'));
```

## Export
Extractors that implement the Exporter interface have an additional `export()` method that takes an iterable type and exports the data to storage.

```php
public export(iterable $iterator, ?array $header = null) : void
```

```php
$extractor->export($dataset);
```

!!! note
    The extractor will overwrite any existing data if the file or database already exists.

## Return an Iterator
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
