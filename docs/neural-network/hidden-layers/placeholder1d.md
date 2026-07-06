<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Layers/Placeholder1D.php">[source]</a></span>

# Placeholder 1D

The Placeholder 1D input layer represents the future input values of a mini batch (matrix) of single dimensional tensors (vectors) to the neural network. It performs shape validation on the input and then forwards it unchanged to the next layer.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | inputs | | int | The number of input nodes (features). |

## Example
```php
use Rubix\ML\NeuralNet\Layers\Placeholder1D;

$layer = new Placeholder1D(10);
```
