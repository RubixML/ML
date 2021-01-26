<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/CostFunctions/CrossEntropy.php">[source]</a></span>

# Cross Entropy
Cross Entropy (or *log loss*) measures the performance of a classification model whose output is a joint probability distribution over the possible classes. Entropy increases as the predicted probability distribution diverges from the actual distribution.

$$
Cross Entropy = -\sum_{c=1}^My_{o,c}\log(p_{o,c})
$$

## Parameters
This cost function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;

$costFunction = new CrossEntropy();
```