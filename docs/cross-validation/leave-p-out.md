<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/LeavePOut.php">[source]</a></span>

# Leave P Out
Leave P Out tests a learner with a unique holdout set of size p for each iteration until all samples have been tested. Although Leave P Out can take long with large datasets and small values of p, it is especially suited for small datasets.

**Interfaces:** [Validator](api.md#validator), [Parallel](#parallel)

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | p | 10 | int | The number of samples to leave out each round for testing. |

## Example
```php
use Rubix\ML\CrossValidation\LeavePOut;

$validator = new LeavePOut(50);
```