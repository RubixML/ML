<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/CostFunctions/CrossEntropy.php">Source</a></span></p>

# Cross Entropy
Cross Entropy, or *log loss*, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect score would have a log loss of 0.

### Parameters
This cost function does not have any parameters.

### Example
```php
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;

$costFunction = new CrossEntropy();
```