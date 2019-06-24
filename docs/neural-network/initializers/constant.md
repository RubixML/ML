### Constant
Initialize the parameter to a user specified constant value.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/Constant.php)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | value | 0. | float | The value to initialize the parameter to. |

**Example:**

```php
use Rubix\ML\NeuralNet\Initializers\Constant;

$initializer = new Constant(1.0);
```