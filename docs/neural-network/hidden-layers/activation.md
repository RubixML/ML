<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Activation.php">[source]</a></span>

# Activation
Activation layers apply a user-defined non-linear activation function to their inputs. They often work in conjunction with [Dense](https://docs.rubixml.com/en/latest/neural-network/hidden-layers/dense.html) layers as a way to transform their output.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | activation fn | | ActivationFunction | The function that computes the output of the layer. |

## Example
```php
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;

$layer = new Activation(new ReLU());
```