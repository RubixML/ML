<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/CostFunctions/MulticlassCrossEntropy.php">[source]</a></span>

# Multiclass Cross Entropy
Multiclass Cross Entropy measures the performance of a multiclass classification model whose output is a probability distribution over the possible classes. Cross-entropy loss increases as the predicted probability distribution diverges from the actual distribution.

$$
Multiclass\ Cross\ Entropy = -\frac{1}{N}\sum_{i=1}^N\sum_{c=1}^C y_{i,c}\log(p_{i,c})
$$

## Parameters
This cost function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;

$costFunction = new MulticlassCrossEntropy();
```
