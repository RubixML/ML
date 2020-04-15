<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/MultilayerPerceptron.php">[source]</a></span>

# Multilayer Perceptron
A multiclass feed forward neural network classifier with user-defined hidden layers. The Multilayer Perceptron is a deep learning model capable of forming higher-order feature representations through layers of computation. In addition, the MLP features progress monitoring which stops training when it can no longer make progress. It utilizes network snapshotting to make sure that it always has the best model parameters even if progress began to decline during training.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | hidden | | array | An array composing the user-specified hidden layers of the network in order. |
| 2 | batch size | 128 | int | The number of training samples to process at a time. |
| 3 | optimizer | Adam | Optimizer | The gradient descent optimizer used to update the network parameters. |
| 4 | alpha | 1e-4 | float | The amount of L2 regularization applied to the weights of the output layer. |
| 5 | epochs | 1000 | int | The maximum number of training epochs. i.e. the number of times to iterate over the entire training set before terminating. |
| 6 | min change | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 7 | window | 3 | int | The number of epochs without improvement in the validation score to wait before considering an early stop. |
| 8 | holdout | 0.1 | float | The proportion of training samples to use for validation and progress monitoring. |
| 9 | cost fn | CrossEntropy | ClassificationLoss | The function that computes the loss associated with an erroneous activation during training. |
| 10 | metric | FBeta | Metric | The validation metric used to score the generalization performance of the model during training. |

## Additional Methods
Return the training loss at each epoch:
```php
public steps() : array
```

Return the validation scores at each epoch:
```php
public scores() : array
```

Returns the underlying neural network instance or `null` if untrained:
```php
public network() : Network|null
```

## Example
```php
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\CrossValidation\Metrics\MCC;

$estimator = new MultilayerPerceptron([
    new Dense(200),
    new Activation(new LeakyReLU()),
    new Dropout(0.3),
    new Dense(100),
    new Activation(new LeakyReLU()),
    new Dropout(0.3),
    new Dense(50),
    new PReLU(),
], 128, new Adam(0.001), 1e-4, 1000, 1e-3, 3, 0.1, new CrossEntropy(), new MCC());
```

### References
>- G. E. Hinton. (1989). Connectionist learning procedures.
>- L. Prechelt. (1997). Early Stopping - but when?