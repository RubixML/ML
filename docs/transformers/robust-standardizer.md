<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/RobustStandardizer.php">[source]</a></span>

# Robust Standardizer
This standardizer transforms continuous features by centering them around the median and scaling by the median absolute deviation (MAD). The use of robust statistics make this standardizer more immune to outliers than the [Z Scale Standardizer](#z-scale-standardizer) which used mean and variance.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Continuous

## Parameters
| # | Param | Default | Type | Description | 
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
