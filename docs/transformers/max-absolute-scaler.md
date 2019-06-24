<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/MaxAbsoluteScaler.php">Source</a></span></p>

# Max Absolute Scaler
Scale the sample matrix by the maximum absolute value of each feature column independently such that the feature will be between -1 and 1.

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful), [Elastic](#elastic)

**Data Type Compatibility:** Continuous

### Parameters
This transformer does not have any parameters.

### Additional Methods
Return the maximum absolute values for each feature column:
```php
public maxabs() : array
```

### Example
```php
use Rubix\ML\Transformers\MaxAbsoluteScaler;

$transformer = new MaxAbsoluteScaler();
```