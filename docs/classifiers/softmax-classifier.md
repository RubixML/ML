<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/SoftmaxClassifier.php">[source]</a></span>

# Softmax Classifier
A multiclass generalization of [Logistic Regression](logistic-regression.md) using a single layer neural network with a [Softmax](../neural-network/activation-functions/softmax.md) output layer.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | batchSize | 256 | int | The number of training samples to process at a time. |
| 2 | optimizer | Adam | Optimizer | The gradient descent optimizer used to update the network parameters. |
| 3 | alpha | 1e-4 | float | The amount of L2 regularization applied to the weights of the output layer. |
| 4 | epochs | 1000 | int | The maximum number of training epochs. i.e. the number of times to iterate over the entire training set before terminating. |
| 5 | minChange | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 6 | window | 5 | int | The number of epochs without improvement in the training loss to wait before considering an early stop. |
| 7 | costFn | CrossEntropy | ClassificationLoss | The function that computes the loss associated with an erroneous activation during training. |

## Example
```php
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;

$estimator = new SoftmaxClassifier(256, new Momentum(0.001), 1e-4, 300, 1e-4, 10, new CrossEntropy());
```

## Additional Methods
Return the loss at each epoch from the last training session:
```php
public steps() : float[]|null
```

Return the underlying neural network instance or `null` if untrained:
```php
public network() : Network|null
```
