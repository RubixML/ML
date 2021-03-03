<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Initializers/Xavier1.php">[source]</a></span>

# Xavier 1
The Xavier 1 initializer draws from a uniform distribution [-limit, limit] where *limit* is equal to sqrt(6 / (fanIn + fanOut)). This initializer is best suited for layers that feed into an activation layer that outputs a value between 0 and 1 such as [Softmax](../activation-functions/softmax.md) or [Sigmoid](../activation-functions/sigmoid.md).

## Parameters
This initializer does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\Initializers\Xavier1;

$initializer = new Xavier1();
```

## References
[^1]: X. Glorot et al. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks.