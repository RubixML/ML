# Extractors
Extractors are data table iterators that help you import data from various source formats such as CSV, JSON, and NDJSON in an efficient way. They implement one of the standard PHP [Traversable](https://www.php.net/manual/en/class.traversable.php) interfaces and can be used to instantiate a new [Dataset](datasets/api.md) object by passing it to the `fromIterator()` method.

**Example**

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;

$dataset = Labeled::fromIterator(new NDJSON('example.ndjson'));
```

Extractors can also be used on their own to loop through a data table. In the example below, we iterate over the records of the NDJSON file, pick out the label, and use the rest of the columns for the sample data.

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