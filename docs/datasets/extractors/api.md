# Extractors
Dataset Extractor objects help you import data from common source formats such as CSV, NDJSON, and SQL. They provide the `extract()` method that allows you to cursor over the data table and returns a ready-to-go [Dataset](../api.md) object.

### Extract an Unlabeled Dataset
Extract and build an unlabeled dataset object from source:
```php
public extract(int $offset = 0, ?int $limit = null) : Unlabeled
```

### Extract a Labeled Dataset
Extract and build a labeled dataset object from source:
```php
public extractWithLabels(int $offset = 0, ?int $limit = null) : Labeled
```

**Example**

```php
use Rubix\ML\Datasets\Extractors\NDJSON;

$extractor = new NDJSON('example.ndjson');

$dataset = $extractor->extract(0, 5000); // Extract first 5000 rows

$dataset = $extractor->extractWithLabels(); // Extract all rows with labels
```