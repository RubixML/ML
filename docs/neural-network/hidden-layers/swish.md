<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Layers/Swish.php">[source]</a></span>

# Swish
Swish is a parametric activation layer that utilizes smooth rectified activation functions. The trainable *beta* parameter allows each activation function in the layer to tailor its output to the training set by interpolating between the linear function and ReLU.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | initializer | Constant | Initializer | The initializer of the beta parameter. |

## Example
```php
use Rubix\ML\NeuralNet\Layers\Swish;
use Rubix\ML\NeuralNet\Initializers\Constant;

$layer = new Swish(new Constant(1.0));
```

## References
[^1]: P. Ramachandran er al. (2017). Swish: A Self-gated Activation Function.
[^2]: P. Ramachandran et al. (2017). Searching for Activation Functions.
