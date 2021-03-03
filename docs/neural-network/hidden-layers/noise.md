<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Layers/Noise.php">[source]</a></span>

# Noise
This layer adds random Gaussian noise to the inputs with a user-defined standard deviation. Noise added to neural network activations acts as a regularizer by indirectly adding a penalty to the weights through the cost function in the output layer.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | stddev | 0.1 | float | The standard deviation of the Gaussian noise added to the inputs. |

## Example
```php
use Rubix\ML\NeuralNet\Layers\Noise;

$layer = new Noise(1e-3);
```

## References
[^1]: C. Gulcehre et al. (2016). Noisy Activation Functions.