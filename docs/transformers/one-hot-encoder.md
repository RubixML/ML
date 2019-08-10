<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/OneHotEncoder.php">Source</a></span>

# One Hot Encoder
The One Hot Encoder takes a feature column of categorical values and produces an n-d *one-hot* representation where n is equal to the number of unique categories in that column. After the transformation, a 0 in any location indicates that the category represented by that column is not present in the sample whereas a 1 indicates that a category is present. One hot encoded is typically used to convert categorical data to continuous so that it can be used to train a learner that is only compatible with continuous features.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Categorical

### Parameters
This transformer does not have any parameters.

### Additional Methods
This transformer does not have any additional methods.

### Example
```php
use Rubix\ML\Transformers\OneHotEncoder;

$transformer = new OneHotEncoder();
```