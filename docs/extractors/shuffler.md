<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Extractors/Shuffler.php">[source]</a></span>

# Shuffler

Randomizes the order of the records of a data table while they are in flight. Shuffler uses a bounded shuffle buffer under the hood to produce a random ordering of the records without holding the entire stream in memory at once.

!!! note
    When the number of records in the stream is greater than the buffer size, the resulting order is not guaranteed to be uniformly random since doing so would require holding the entire stream in memory at once.

**Interfaces:** [Extractor](api.md)

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | iterator | | Traversable | The base iterator. |
| 2 | bufferSize | 256 | int | The maximum number of records to hold in memory at a time. |

## Example

```php
use Rubix\ML\Extractors\Shuffler;
use Rubix\ML\Extractors\NDJSON;

$extractor = new Shuffler(new NDJSON('example.jsonl'), bufferSize: 1024);
```

## Additional Methods

This extractor does not have any additional methods.

## Reproducibility

The shuffle uses the globally seeded RNG and can be reproduced by seeding PHP's randomness before iterating.

```php
use Rubix\ML\Extractors\Shuffler;
use Rubix\ML\Extractors\CSV;

srand(1234);

$extractor = new Shuffler(new CSV('example.csv', true), bufferSize: 1024);

foreach ($extractor as $record) {
    // ...
}
```
