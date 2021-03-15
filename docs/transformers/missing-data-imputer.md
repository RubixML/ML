<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/MissingDataImputer.php">[source]</a></span>

# Missing Data Imputer
Missing Data Imputer replaces missing continuous (denoted by `NaN`) or categorical values (denoted by special placeholder category such as `'?'`) with a guess based on user-defined Strategy.

**Interfaces:** [Transformer](api.md#transformers), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical and Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | continuous | Mean | Strategy | The guessing strategy to employ for continuous feature columns. |
| 2 | categorical | K Most Frequent | Strategy | The guessing strategy to employ for categorical feature columns. |
| 3 | categoricalPlaceholder | '?' | string | The special placeholder category that denotes missing values. |

## Example
```php
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Strategies\Percentile;
use Rubix\ML\Strategies\Prior;

$transformer = new MissingDataImputer(new Percentile(0.55), new Prior(), '?');
```

## Additional Methods
This transformer does not have any additional methods.
