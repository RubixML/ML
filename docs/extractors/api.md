# Extractors
Extractors are data table iterators that help you import data from various source formats such as CSV, JSON, and NDJSON in an efficient way. They implement one of the standard PHP [Traversable](https://www.php.net/manual/en/class.traversable.php) interfaces and can be used to instantiate a new [Dataset](datasets/api.md) object by passing it to the `fromIterator()` method.

**Example**

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;

$dataset = Labeled::fromIterator(new NDJSON('example.ndjson'));
```

Extractors can also be used on their own to loop through a data table. In the example below, we iterate over the records of the NDJSON file, pick out the label, and use the rest of the columns for the sample data.

> **Note:** Extractors are read-only, they will never overwrite the source dataset.

**Example**

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;

$extractor = new NDJSON('example.ndjson');

$samples = $labels = [];

foreach ($extractor as $record) {
    $labels[] = $record['class'];

    unset($record['class']);

    $samples[] = $record;
}

$dataset = new Labeled($samples, $labels);
```

### Return the Wrapped Iterator
To return the underlying iterator wrapped by an extractor:
```php
public getIterator() : Traversable
```

The example below shows how you can instantiate a new dataset object using only a portion of the source dataset by wrapping an underlying iterator with the standard PHP library's [Limit Iterator](https://www.php.net/manual/en/class.limititerator.php). For this example, we'll choose to load 1,000 rows of data after the first 500 rows.

**Example**

```php
use Rubix\ML\Datasets\Unlabeled;
use LimitIterator;

$iterator = new LimitIterator($extractor->getIterator(), 500, 1000);

$dataset = Unlabeled::fromIterator($iterator);
```