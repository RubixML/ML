<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Layers/Dropout.php">Source</a></span>

# Dropout
Dropout layers temporarily disable neuron activations during each training pass. It is a regularization and model averaging technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | ratio | 0.5 | float | The ratio of neurons that are dropped during each training pass. |

### Example
```php
use Rubix\ML\NeuralNet\Layers\Dropout;

$layer = new Dropout(0.5);
```

### References
>- N. Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.