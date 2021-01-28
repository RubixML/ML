<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Layers/Activation.php">[source]</a></span>

# Activation
Activation layers apply a user-defined non-linear activation function to their inputs. They often work in conjunction with [Dense](dense.md) layers as a way to transform their output.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | activationFn | | ActivationFunction | The function that computes the output of the layer. |

## Example
```php
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;

$layer = new Activation(new ReLU());
```