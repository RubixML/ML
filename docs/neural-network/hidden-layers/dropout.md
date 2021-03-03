<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Layers/Dropout.php">[source]</a></span>

# Dropout
Dropout is a regularization technique to reduce overfitting in neural networks by preventing complex co-adaptations on training data. It works by temporarily disabling output nodes during each training pass. It also acts as an efficient way of performing model averaging with the parameters of neural networks.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | ratio | 0.5 | float | The ratio of nodes that are dropped during each training pass. |

## Example
```php
use Rubix\ML\NeuralNet\Layers\Dropout;

$layer = new Dropout(0.2);
```

## References
[^1]: N. Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.