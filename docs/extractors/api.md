# Extractors
Data Extractors help you import data from various source formats such as CSV, NDJSON, and SQL. They can be used to instantiate a new [Dataset](datasets/api.md) object by passing it to the `fromIterator()` method or they can be used on their own to generate an iterable data table.

### Extract a Data Table
Read the records and return them in an iterator:
```php
public extract() : iterable
```

### Cursoring
Extractors that extend the Cursor class are able to iterate over portions of the dataset without loading it all into memory. They have the `setOffset()` and `setLimit()` methods that allow you to control the window of data.

Set the line offset of the cursor:
```php
public setOffset(int $offset) : self
```

Set the maximum number of rows of data to return:
```php
public setLimit(int $limit) : self
```

**Examples**

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;

$dataset = Labeled::fromIterator(new NDJSON('example.ndjson'));
```

```php
$extractor = new NDJSON('example.ndjson');

$extractor->setOffset(100)->setLimit(5000); // Set the cursor

foreach ($extractor as $key => $record) {
    // ...
}
```