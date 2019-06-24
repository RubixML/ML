### Relative Entropy
Relative Entropy or *Kullback-Leibler divergence* is a measure of how the expectation and activation of the network diverge.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/CostFunctions/RelativeEntropy.php)

**Parameters:**

This cost function does not have any parameters.

**Example:**

```php
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;

$costFunction = new RelativeEntropy();
```