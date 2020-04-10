<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Dense.php">[source]</a></span>

# Dense
Dense (or *fully connected*) hidden layers are layers of neurons that connect to each node in the previous layer by a parameterized synapse. They perform a linear transformation on their input and are usually followed by an [Activation](activation.md) layer. The majority of the trainable parameters in a standard feed forward neural network are contained within Dense hidden layers.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | neurons | | int | The number of nodes in the layer. |
| 2 | alpha | 0.0 | float | The amount of L2 regularization applied to the weights. |
| 3 | bias | true | bool | Should the layer include a bias parameter? |
| 4 | weight initializer | He | Initializer | The initializer of the weight parameter. |
| 5 | bias initializer | Constant | Initializer | The initializer of the bias parameter. |

## Example
```php
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Initializers\Constant;

$layer = new Dense(100, 1e-4, true, new He(), new Constant(0.0));
```