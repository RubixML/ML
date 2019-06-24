### Alpha Dropout
Alpha Dropout is a type of dropout layer that maintains the mean and variance of the original inputs in order to ensure the self-normalizing property of [SELU](#selu) networks with dropout. Alpha Dropout fits with SELU networks by randomly setting activations to the negative saturation value of the activation function at a given ratio each pass.

> **Note**: Alpha Dropout is generally only used in the context of SELU networks. Use regular [Dropout](#dropout) for other types of neural nets.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/AlphaDropout.php)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | ratio | 0.1 | float | The ratio of neurons that are dropped during each training pass. |

**Example:**

```php
use Rubix\ML\NeuralNet\Layers\AlphaDropout;

$layer = new AlphaDropout(0.1);
```

**References:**

>- G. Klambauer et al. (2017). Self-Normalizing Neural Networks.