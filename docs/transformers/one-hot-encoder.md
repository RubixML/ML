<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/OneHotEncoder.php">[source]</a></span>

# One Hot Encoder
The One Hot Encoder takes a categorical feature column and produces an n-dimensional continuous representation where *n* is equal to the number of unique categories present in that column. A `0` in any location indicates that the category represented by that column is not present in the sample, whereas a `1` indicates that a category is present.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical

## Parameters
This transformer does not have any parameters.

## Example
```php
use Rubix\ML\Transformers\OneHotEncoder;

$transformer = new OneHotEncoder();
```

## Additional Methods
Return the categories computed during fitting indexed by feature column:
```php
public categories() : ?array
```
