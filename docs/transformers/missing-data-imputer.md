<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/MissingDataImputer.php">Source</a></span></p>

# Missing Data Imputer
In the real world, it is common to have data with missing values here and there. The Missing Data Imputer replaces missing value *placeholder* values with a guess based on a given [Strategy](#guessing-strategies).

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

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
use Rubix\ML\Other\Strategies\BlurryPercentile;
use Rubix\ML\Other\Strategies\PopularityContest;

$transformer = new MissingDataImputer('?', new BlurryPercentile(0.61), new PopularityContest());
```