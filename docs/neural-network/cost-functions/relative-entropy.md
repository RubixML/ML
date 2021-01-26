<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/CostFunctions/RelativeEntropy.php">[source]</a></span>

# Relative Entropy
Relative Entropy (or *Kullback-Leibler divergence*) is a measure of how the expectation and activation of the network diverge. It is different from [Cross Entropy](cross-entropy.md) in that it is *asymmetric* and thus does not qualify as a statistical measure of error.

$$
KL(\hat{y} || y) = \sum_{c=1}^{M}\hat{y}_c \log{\frac{\hat{y}_c}{y_c}}
$$

## Parameters
This cost function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;

$costFunction = new RelativeEntropy();
```