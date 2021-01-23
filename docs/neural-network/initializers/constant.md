<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Initializers/Constant.php">[source]</a></span>

# Constant
Initialize the parameter to a user-specified constant value.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | value | 0.0 | float | The value to initialize the parameter to. |

## Example
```php
use Rubix\ML\NeuralNet\Initializers\Constant;

$initializer = new Constant(1.0);
```