<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Initializers/Uniform.php">[source]</a></span>

# Uniform
Generates a random uniform distribution centered at 0 and bounded at both ends by the parameter beta.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | beta | 0.05 | float | The upper and lower bound of the distribution. |

## Example
```php
use Rubix\ML\NeuralNet\Initializers\Uniform;

$initializer = new Uniform(1e-3);
```