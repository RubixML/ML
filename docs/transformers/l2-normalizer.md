<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/L2Normalizer.php">Source</a></span></p>

# L2 Normalizer
Transform each sample vector in the sample matrix such that each feature is divided by the L2 norm (or *magnitude*) of that vector.

**Interfaces:** [Transformer](#transformers)

**Data Type Compatibility:** Continuous only

### Parameters
This transformer does not have any parameters.

### Additional Methods
This transformer does not have any additional methods.

### Example
```php
use Rubix\ML\Transformers\L2Normalizer;

$transformer = new L2Normalizer();
```