<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/CostFunctions/CrossEntropy.php">[source]</a></span>

# Cross Entropy
Cross Entropy (or *log loss*) measures the performance of a classification model whose output is a joint probability distribution over the possible classes. Entropy increases as the predicted probability distribution diverges from the actual distribution.

## Parameters
This cost function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;

$costFunction = new CrossEntropy();
```