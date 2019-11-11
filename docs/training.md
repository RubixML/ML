# Training
Most estimators in Rubix ML need to be trained before they can make predictions. Estimators that require training are called Learners and implement the `train()` method of the [Learner](learner.md) interface among others. Training is the process of feeding data to the learner so that it can build an internal represenation of the problem space. For example, some models such as the ones produced by [K Nearest Neighbors](./classifiers/k-nearest-neighbors.md) consider each sample to be a point in high-dimensional Euclidean space. Neural network models, on the other hand, consider samples to be the impulses applied to a complex interconnected network of neurons and synapses. No matter *how* the learner works under the hood the API is the same.

**Example**

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\KNearestNeighbors;

// Import samples and labels

$dataset = new Labeled($samples, $labels);

$estimator = new KNearestNeighbors(10);

$estimator->train($dataset);
```

### Batch vs Online Learning
Batch learning is when a learner is trained in full within a single session using only one dataset. Calling the `train()` method on the learner instance is an example of batch learning. In contrast, Online learning occurs when a learner is trained over multiple sessions with datasets as small as a single sample each. Learners that are capable of being partially trained in Rubix ML implement the [Online](online.md) interface that includes the `partial()` method to train in online mode. Subsequent calls the to `partial()` method will continue training where the learner had left off since the last training session.

**Example**

```php
$folds = $dataset->fold(3);

$estimator->partial($folds[0]);

$estimator->partial($folds[1]);

$estimator->partial($folds[2]);
```
