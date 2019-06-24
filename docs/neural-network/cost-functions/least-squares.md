### Least Squares
Least Squares or *quadratic* loss is a function that measures the squared error between the target output and the actual output of the network.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/CostFunctions/LeastSquares.php)

**Parameters:**

This cost function does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;

$costFunction = new LeastSquares();
```