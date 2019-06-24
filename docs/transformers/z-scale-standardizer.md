### Z Scale Standardizer
A method of centering and scaling a dataset such that it has 0 mean and unit variance, also known as a Z Score.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/ZScaleStandardizer.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful), [Elastic](#elastic)

**Compatibility:** Continuous

**Parameters:**
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | center | true | bool | Should we center the sample dataset? |

**Additional Methods:**

Return the means calculated by fitting the training set:
```php
public means() : array
```

Return the variances calculated during fitting:
```php
public variances() : array
```

Return the standard deviations calculated during fitting:
```php
public stddevs() : array
```

**Example:**

```php
use Rubix\ML\Transformers\ZScaleStandardizer;

$transformer = new ZScaleStandardizer(true);
```

**References:**

>- T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for Computing Sample Variances.