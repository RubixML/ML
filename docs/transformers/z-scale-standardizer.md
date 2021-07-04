<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/ZScaleStandardizer.php">[source]</a></span>

# Z Scale Standardizer
A method of centering and scaling a dataset such that it has 0 mean and unit variance, also known as a Z-Score. Although Z-Scores are technically unbounded, in practice they mostly fall between -3 and 3 - that is, they are no more than 3 standard deviations away from the mean.

$$
{\displaystyle z = {x - \mu \over \sigma }}
$$

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Elastic](api.md#elastic), [Reversible](api.md#reversible), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | center | true | bool | Should we center the data at 0? |

## Example
```php
use Rubix\ML\Transformers\ZScaleStandardizer;

$transformer = new ZScaleStandardizer(true);
```

## Additional Methods
Return the means calculated by fitting the training set:
```php
public means() : array
```

Return the variances calculated during fitting:
```php
public variances() : array
```

## References
[^1]: T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for Computing Sample Variances.
