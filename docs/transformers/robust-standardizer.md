<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/RobustStandardizer.php">[source]</a></span>

# Robust Standardizer
This standardizer transforms continuous features by centering them around the median and scaling by the median absolute deviation (MAD) referred to as a *robust*  or *modified* Z-Score. The use of robust statistics make this standardizer more immune to outliers than [Z Scale Standardizer](#z-scale-standardizer).

$$
{\displaystyle z^\prime = {x - \operatorname {median}(X) \over MAD }}
$$

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Reversible](api.md#reversible), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description | 
|---|---|---|---|---|
| 1 | center | true | bool | Should we center the data at 0? |

## Example
```php
use Rubix\ML\Transformers\RobustStandardizer;

$transformer = new RobustStandardizer(true);
```

## Additional Methods
Return the medians calculated by fitting the training set:
```php
public medians() : array
```

Return the median absolute deviations calculated during fitting:
```php
public mads() : array
```
