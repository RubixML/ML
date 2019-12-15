# Extractors
Data Extractors help you import data from various source formats such as CSV, NDJSON, and SQL. They can be used to instantiate a new [Dataset](../api.md) object by passing it to the `from()` method or they can be used on their own.

### Extract a Data Table
Read the records and return them in an iterator:
```php
public extract() : iterable
```

### Cursoring
Set the row offset of the cursor:
```php
public setOffset(int $offset) : self
```

Set the maximum number of rows to return:
```php
public setLimit(int $limit) : self
```

**Example**

```php
use Rubix\ML\Datasets\Extractors\NDJSONArray;

$extractor = new NDJSONArray('example.ndjson');

$records = $extractor->setOffset(100)->setLimit(5000)->extract();

foreach ($records as $record) {
    // ...
}
```