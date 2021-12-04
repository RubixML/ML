<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Extractors/Deduplicator.php">[source]</a></span>

# Deduplicator
Removes duplicate records from a dataset while the records are in flight. Deduplicator uses a Bloom filter under the hood to probabilistically identify records that have already been seen before.

!!! note
    Due to its probabilistic nature, Deduplicator may mistakenly drop unique records at a bounded rate.

**Interfaces:** [Extractor](api.md)

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | iterator | | Traversable | The base iterator. |
| 2 | maxFalsePositiveRate | 0.001 | float | The false positive rate to remain below. |
| 3 | numHashes | 4 | int | The number of hash functions used, i.e. the number of slices per layer. Set to null for auto. |
| 4 | layerSize | 32000000 | int | The size of each layer of the filter in bits. |

## Example
```php
use Rubix\ML\Extractors\Deduplicator;
use Rubix\ML\Extractors\CSV;

$extractor = new Deduplicator(new CSV('example.csv', true), 0.01, 3, 32000000);
```

## Additional Methods
Return the number of records that have been dropped so far.
```php
public dropped() : int
```
