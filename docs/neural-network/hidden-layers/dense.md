<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Dense.php">[source]</a></span>

# Dense
Dense layers (or *fully connected* layers) are layers of neurons that connect to each node in the previous layer by a parameterized synapse. The majority of the trainable parameters in a standard feed forward neural network are contained within Dense layers.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | neurons | | int | The number of neurons in the layer. |
| 2 | weight initializer | He | Initializer | The initializer of the weight parameter. |
| 3 | bias initializer | Constant | Initializer | The initializer of the bias parameter. |

## Example
```php
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Initializers\Constant;

$layer = new Dense(100, new He(), new Constant(0.0));
```