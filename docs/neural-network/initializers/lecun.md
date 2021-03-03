<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Initializers/LeCun.php">[source]</a></span>

# Le Cun
Proposed by Yan Le Cun in a paper in 1998, this initializer was one of the first published attempts to control the variance of activations between layers through weight initialization. It remains a good default choice for many hidden layer configurations.

## Parameters
This initializer does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\Initializers\LeCun;

$initializer = new LeCun();
```

## References
[^1]: Y. Le Cun et al. (1998). Efficient Backprop.