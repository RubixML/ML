<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/CostFunctions/RelativeEntropy.php">Source</a></span>

# Relative Entropy
Relative Entropy or *Kullback-Leibler divergence* is a measure of how the expectation and activation of the network diverge. A KL divergence of 0 indicates that two distributions are identical.

### Parameters
This cost function does not have any parameters.

### Example
```php
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;

$costFunction = new RelativeEntropy();
```