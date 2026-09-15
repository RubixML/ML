<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Layers/Dense.php">[source]</a></span>

# Dense

Dense (or *fully connected*) hidden layers are layers of neurons that connect to each node in the previous layer by a parameterized synapse. They perform a linear transformation on their input and are usually followed by an [Activation](activation.md) layer. The majority of the trainable parameters in a standard feed forward neural network are contained within Dense hidden layers. L1 and L2 regularization can be applied to the weights to reduce overfitting; L1 regularization in particular encourages sparsity by driving small weights toward zero.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | neurons | | int | The number of nodes in the layer. |
| 2 | l1Penalty | 0.0 | float | The amount of L1 regularization applied to the weights. |
| 3 | l2Penalty | 0.0 | float | The amount of L2 regularization applied to the weights. |
| 4 | bias | true | bool | Should the layer include a bias parameter? |
| 5 | weightInitializer | He | Initializer | The initializer of the weight parameter. |
| 6 | biasInitializer | Constant | Initializer | The initializer of the bias parameter. |

## Example

```php
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Initializers\Constant;

$layer = new Dense(neurons: 100, l1Penalty: 1e-3, l2Penalty: 1e-4, bias: true, weightInitializer: new He(), biasInitializer: new Constant(0.0));
```
