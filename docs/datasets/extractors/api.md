# Extractors
Dataset Extractor objects help you import data from various source formats such as CSV, NDJSON, and SQL.

### Extract a Dataset
Extract and build a dataset object from source:
```php
public extract(int $offset = 0, ?int $limit = null) : Dataset
```

**Example**

```php
use Rubix\ML\Datasets\Extractors\NDJSON;

$extractor = new NDJSON('example.ndjson');

$dataset = $extractor->extract(0, 5000); // Extract first 5000 rows
```