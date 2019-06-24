### L2 Normalizer
Transform each sample vector in the sample matrix such that each feature is divided by the L2 norm (or *magnitude*) of that vector.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/L2Normalizer.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Continuous only

**Parameters:**

This transformer does not have any parameters.

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\L2Normalizer;

$transformer = new L2Normalizer();
```