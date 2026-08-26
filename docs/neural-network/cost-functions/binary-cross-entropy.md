<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/CostFunctions/BinaryCrossEntropy.php">[source]</a></span>

# Binary Cross Entropy
Binary Cross Entropy (or *log loss*) measures the performance of a binary classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect score would have a log loss of 0.

$$
Binary\ Cross\ Entropy = -\frac{1}{N}\sum_{i=1}^N[y_i\log(p_i) + (1-y_i)\log(1-p_i)]
$$

## Parameters
This cost function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;

$costFunction = new BinaryCrossEntropy();
```
