# Extractors
Extractors are objects that help you import data from various source formats such as CSV, JSON, and NDJSON in an efficient way. They implement one of the standard PHP [Traversable](https://www.php.net/manual/en/class.traversable.php) interfaces and can be used to instantiate a new [Dataset](datasets/api.md) object by passing it to the `fromIterator()` method or they can be used on their own to iterate over a data table.

**Examples**

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;

$dataset = Labeled::fromIterator(new NDJSON('example.ndjson'));
```

```php
$extractor = new NDJSON('example.ndjson');

foreach ($extractor as $key => $record) {
    // ...
}
```