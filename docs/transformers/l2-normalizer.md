<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/L2Normalizer.php">[source]</a></span>

# L2 Normalizer
Transform each sample (row) vector in the sample matrix such that each feature is divided by the L2 norm (or *magnitude*) of that vector.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Continuous only

## Parameters
This transformer does not have any parameters.

## Example
```php
use Rubix\ML\Transformers\L2Normalizer;

$transformer = new L2Normalizer();
```

## Additional Methods
This transformer does not have any additional methods.
