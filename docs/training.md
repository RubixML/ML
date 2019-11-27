# Training
Most estimators in Rubix ML need to be trained before they can make predictions. Estimators that require training are called [Learners](learner.md) and they implement the `train()` method among others. Training is the process of feeding data to the learner so that it can build an internal represenation (or *model*) of the problem space. For example, some models such as the ones produced by [K Nearest Neighbors](./classifiers/k-nearest-neighbors.md) consider each sample to be a point in high-dimensional Euclidean space. Neural network models, on the other hand, consider samples to be the inputs to a complex interconnected network of neurons and synapses. No matter *how* the learner works under the hood the training API is still the same.

**Example**

```php
use Rubix\ML\Classifiers\KNearestNeighbors;

// Import labeled training set

$estimator = new KNearestNeighbors(10);

$estimator->train($dataset);
```

## Batch vs Online Learning
Batch learning is when a learner is trained in full within a single session using only one dataset. Calling the `train()` method on the learner instance is an example of batch learning. In contrast, *online* learning occurs when a learner is trained over multiple sessions with datasets as small as a single sample each. Learners that are capable of being partially trained in Rubix ML implement the [Online](online.md) interface that includes the `partial()` method to train in online mode. Subsequent calls the to `partial()` method will continue training where the learner had left off since the last training session.

**Example**

```php
$folds = $dataset->fold(3);

$estimator->partial($folds[0]);

$estimator->partial($folds[1]);

$estimator->partial($folds[2]);
```

## Monitoring Progress
Since training is often executed in an iterative process, it is sometimes useful to obtain real-time feedback as to how the learner is progressing during training. For example, you may want to monitor the training loss to make sure that it isn't *increasing* instead of decreasing with training. Such early feedback can indicate various pathologies such as model overfitting or improperly tuned hyper-parameters. Learners that implement the [Verbose](verbose.md) interface accept a PSR-3 logger instance that can be used to output training information at each time step (or *epoch*). Rubix ML comes built-in with a [Screen Logger](other/loggers/screen.md) that does the job for most cases.

**Example**

```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('example'));
```

## Ensemble Learning
Some learners are actually collections or *ensembles* of learners that work together to form a single model. Some ensembles such as [Bootstrap Aggregator](bootstrap-aggregator.md) and [Committee Machine](committee-machine.md) work by averaging the predictions of the base estimators. Ensembles such as these are able to produce more stable models than any single member because they reduce the variance of their prediction through averaging. More sophisticated boosting ensembles such as [AdaBoost](classifiers/adaboost.md) and [Gradient Boost](regressors/gradient-boost.md) focus on iteratively improving the predictions of a *weak* learner by using many specialized *weak* learners.

**Examples**

- [Bootstrap Aggregator](bootstrap-aggregator.md)
- [Committee Machine](committee-machine.md)
- [AdaBoost](classifiers/adaboost.md)
- [Random Forest](classifiers/random-forest.md)
- [Isolation Forest](anomaly-detectors/isolation-forest.md)
- [Gradient Boost](regressors/gradient-boost.md)