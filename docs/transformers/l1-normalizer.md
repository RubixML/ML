<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/L1Normalizer.php">[source]</a></span>

# L1 Normalizer
Transform each sample vector in the sample matrix such that each feature is divided by the L1 norm (or *magnitude*) of that vector. The resulting sample will have continuous features between 0 and 1.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Continuous only

## Parameters
This transformer does not have any parameters.

## Additional Methods
This transformer does not have any additional methods.

## Example
```php
use Rubix\ML\Transformers\L1Normalizer;

$transformer = new L1Normalizer();
```