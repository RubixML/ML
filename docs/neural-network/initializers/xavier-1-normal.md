<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Initializers/Xavier1Normal.php">[source]</a></span>

# Xavier 1 Normal
The Xavier 1 Normal initializer draws from a truncated normal distribution with mean 0 and standard deviation equal to sqrt(2 / (fanIn + fanOut)). This initializer is best suited for layers that feed into an activation layer that outputs a value between 0 and 1 such as [Softmax](../activation-functions/softmax.md) or [Sigmoid](../activation-functions/sigmoid.md).

## Parameters
This initializer does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\Initializers\Xavier1Normal;

$initializer = new Xavier1Normal();
```

## References
[^1]: X. Glorot et al. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks.
