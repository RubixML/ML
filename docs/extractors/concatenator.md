<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Extractors/Concatenator.php">[source]</a></span>

# Concatenator
Combines multiple iterators by concatenating the output of one iterator with the output of the next iterator in the series.

**Interfaces:** [Extractor](api.md)
## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | iterators | | iterable | The iterators to concatenate together. |

## Example
```php
use Rubix\ML\Extractors\Concatenator;
use Rubix\ML\Extractors\CSV;

$extractor = new Concatenator([
    new CSV('dataset1.csv'),
    new CSV('dataset2.csv'),
    new CSV('dataset3.csv'),
]);
```

## Additional Methods
This extractor does not have any additional methods.
