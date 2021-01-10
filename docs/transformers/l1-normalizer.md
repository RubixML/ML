<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/L1Normalizer.php">[source]</a></span>

# L1 Normalizer
Transform each sample (row) vector in the sample matrix such that each feature is divided by the L1 norm (or *magnitude*) of that vector.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Continuous only

## Parameters
This transformer does not have any parameters.

## Example
```php
use Rubix\ML\Transformers\L1Normalizer;

$transformer = new L1Normalizer();
```

## Additional Methods
This transformer does not have any additional methods.
