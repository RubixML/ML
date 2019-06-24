### Uniform
Generates a random uniform distribution centered at 0 and bounded at both ends by the parameter beta.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Initializers/Uniform.php)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | beta | 0.05 | float | The minimum and maximum bound on the random distribution. |

**Example:**

```php
use Rubix\ML\NeuralNet\Initializers\Uniform;

$initializer = new Uniform(1e-3);
```