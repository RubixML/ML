<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/MissingDataImputer.php">Source</a></span>

# Missing Data Imputer
In the real world, it is common to have data with missing values here and there. The Missing Data Imputer replaces missing value *placeholder* values with a guess based on a given guessing [Strategy](../other/strategies/api.md).

**Interfaces:** [Transformer](api.md#transformers), [Stateful](api.md#stateful)

**Data Type Compatibility:** Categorical, Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | placeholder | '?' | string or numeric | The placeholder value that denotes a missing value. |
| 2 | continuous strategy | Mean | object | The guessing strategy to employ for continuous feature columns. |
| 3 | categorical strategy | K Most Frequent | object | The guessing strategy to employ for categorical feature columns. |

### Additional Methods
This transformer does not have any additional methods.

### Example
```php
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Other\Strategies\Percentile;
use Rubix\ML\Other\Strategies\Prior;

$transformer = new MissingDataImputer('?', new Percentile(0.61), new Prior());
```