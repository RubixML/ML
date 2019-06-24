### One Hot Encoder
The One Hot Encoder takes a column of categorical features and produces a n-d *one-hot* representation where n is equal to the number of unique categories in that column. A 0 in any location indicates that a category represented by that column is not present whereas a 1 indicates that a category is present in the sample.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/OneHotEncoder.php)

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

**Compatibility:** Categorical

**Parameters:**

This transformer does not have any parameters.

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\OneHotEncoder;

$transformer = new OneHotEncoder();
```