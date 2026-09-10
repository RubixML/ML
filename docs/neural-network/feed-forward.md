<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/FeedForward.php">[source]</a></span>

# Feed Forward

The Feed Forward network is the core neural network implementation of the library consisting of an input layer, any number of intermediate hidden layers, and an output layer. The parameters of the network are learned using mini batch gradient descent with backpropagation. It is the network used under the hood by the neural network learners such as [Multilayer Perceptron](../classifiers/multilayer-perceptron.md), [MLP Regressor](../regressors/mlp-regressor.md), [Adaline](../regressors/adaline.md), [Softmax Classifier](../classifiers/softmax-classifier.md), and [Logistic Regression](../classifiers/logistic-regression.md).

!!! note
    The Feed Forward network is part of the neural network subsystem and is not a standalone estimator.

## Example

```php
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;

$network = new FeedForward(
    new Placeholder1D(784),
    [
        new Dense(neurons: 200),
        new Activation(activationFn: new LeakyReLU()),
        new Dense(neurons: 100),
        new Activation(activationFn: new LeakyReLU()),
    ],
    new Multiclass(
        classes: ['cat', 'dog', 'bird'],
        costFn: new MulticlassCrossEntropy()
    )
);
```

## API Reference

Return the input layer of the network:

```php
public function input() : Input
```

Return an array of hidden layers indexed left to right:

```php
public function hidden() : array
```

Return the output layer of the network:

```php
public function output() : Output
```

Return all the layers in the network in the order they are executed:

```php
public function layers() : Traversable
```

Return the total number of parameters in the network:

```php
public function numParams() : int
```

Return an iterable of all the parameters in the network:

```php
public function parameters() : Traversable
```

Return the number of trainable (unfrozen) parameters in the network:

```php
public function numTrainableParams() : int
```

Return an iterable of all the trainable (unfrozen) parameters in the network:

```php
public function trainableParameters() : Traversable
```

Initialize the parameters of the layers. Called once before the first training session to set up the network:

```php
public function initialize() : void
```

Freeze the first `k` hidden layers of the network preventing their parameters from being updated during training. Useful for fine-tuning a pretrained model.

```php
public function freezeFirstKLayers(int $k) : void
```

Unfreeze all hidden layers allowing their parameters to be updated during training.

```php
public function unfreeze() : void
```

Run an inference pass and return the activations at the output layer:

```php
public function infer(Dataset $dataset) : Matrix
```

Perform a forward and backward pass of the network in one call returning the loss from the backward pass:

```php
public function roundtrip(Labeled $dataset) : float
```

Feed a batch through the network and return a matrix of activations at the output layer:

```php
public function feed(Matrix $input) : Matrix
```

Backpropagate the gradient of the cost function and return the loss:

```php
public function backpropagate(array $labels) : float
```

Export the network architecture as a graph in dot format:

```php
public function exportGraphviz() : Encoding
```

```php
use Rubix\ML\Helpers\Graphviz;
use Rubix\ML\Persisters\Filesystem;

$dot = $network->exportGraphviz();

Graphviz::dotToImage($dot)->saveTo(new Filesystem('network.png'));
```
