### Dense
Dense layers are fully connected neuronal layers, meaning each neuron is connected to each other in the previous layer by a weighted *synapse*. The majority of the parameters in a standard feedforward network are usually contained within the Dense hidden layers of the network.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Dense.php)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | neurons | None | int | The number of neurons in the layer. |
| 2 | weight initializer | He | object | The initializer of the weight parameter. |
| 3 | bias initializer | Constant | object | The initializer of the bias parameter. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Initializers\Constant;

$layer = new Dense(100, new He(), new Constant(0.));
```