<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/LogisticRegresion.php">[source]</a></span>

# Logistic Regression
A linear classifier that uses the logistic (*sigmoid*) function to estimate the probabilities of exactly two class outcomes. The model parameters (weights and bias) are solved using Mini Batch Gradient Descent with pluggable optimizers and cost functions that run on the neural network subsystem.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Ranks Features](../ranks-features.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | batchSize | 128 | int | The number of training samples to process at a time. |
| 2 | optimizer | Adam | Optimizer | The gradient descent optimizer used to update the network parameters. |
| 3 | l2Penalty | 1e-4 | float | The amount of L2 regularization applied to the weights of the output layer. |
| 4 | epochs | 1000 | int | The maximum number of training epochs. i.e. the number of times to iterate over the entire training set before terminating. |
| 5 | minChange | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 6 | window | 5 | int | The number of epochs without improvement in the training loss to wait before considering an early stop. |
| 7 | costFn | CrossEntropy | ClassificationLoss | The function that computes the loss associated with an erroneous activation during training. |

## Example
```php
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;

$estimator = new LogisticRegression(64, new Adam(0.001), 1e-4, 100, 1e-4, 5, new CrossEntropy());
```

## Additional Methods
Return an iterable progress table with the steps from the last training session:
```php
public steps() : iterable
```

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->steps());
```

Return the loss for each epoch from the last training session:
```php
public losses() : float[]|null
```

Return the underlying neural network instance or `null` if untrained:
```php
public network() : Network|null
```
