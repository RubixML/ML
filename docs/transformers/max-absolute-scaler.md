<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/MaxAbsoluteScaler.php">[source]</a></span>

# Max Absolute Scaler
Scale the sample matrix by the maximum absolute value of each feature column independently such that the feature value is between -1 and 1.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Elastic](api.md#elastic), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
This transformer does not have any parameters.

## Example
```php
use Rubix\ML\Transformers\MaxAbsoluteScaler;

$transformer = new MaxAbsoluteScaler();
```

## Additional Methods
Return the maximum absolute values for each feature column:
```php
public maxabs() : array
```
