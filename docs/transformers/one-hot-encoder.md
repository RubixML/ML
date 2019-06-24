<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/OneHotEncoder.php">Source</a></span></p>

# One Hot Encoder
The One Hot Encoder takes a column of categorical features and produces a n-d *one-hot* representation where n is equal to the number of unique categories in that column. A 0 in any location indicates that a category represented by that column is not present whereas a 1 indicates that a category is present in the sample.

**Interfaces:** [Transformer](#transformers), [Stateful](#stateful)

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