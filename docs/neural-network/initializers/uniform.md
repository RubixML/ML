<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/Uniform.php">Source</a></span>

# Uniform
Generates a random uniform distribution centered at 0 and bounded at both ends by the parameter beta.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | beta | 0.05 | float | The minimum and maximum bound on the random distribution. |

### Example
```php
use Rubix\ML\NeuralNet\Initializers\Uniform;

$initializer = new Uniform(1e-3);
```