# Training
Most estimators have the ability to be trained using a dataset. Estimators that require training are called [Learners](learner.md) and implement the `train()` method among others. Training is the process of feeding data to a learner so that it can build an internal representation (or *model*) of the problem. Supervised learners require a dataset with labels. Unsupervised learners can be trained with either a labeled or unlabeled dataset but only the samples are used to build a model. Every learner has a unique way of learning but no matter *how* it works under the hood the training API is the same.

To begin training a learner, pass a dataset object to the `train()` method on the learner instance like in the example below.

```php
$estimator->train($dataset);
```

## Batch vs Online Learning
Batch learning is when a learner is trained in full using only one dataset in a single session. Calling the `train()` method on a learner instance is an example of batch learning. In contrast, *online* learning occurs when a learner is trained over multiple sessions with multiple datasets as small as a single sample each. Learners that are capable of being partially trained like this implement the [Online](online.md) interface which includes the `partial()` method for training in an online scheme. Subsequent calls to the `partial()` method will continue training where the learner left off since the last training session. Online learning is especially useful for when you have a dataset that is too large to fit into memory all at once.

```php
$folds = $dataset->fold(3);

$estimator->train($folds[0]);

$estimator->partial($folds[1]);

$estimator->partial($folds[2]);
```

> **Note:** After the initial training, the learner will expect subsequent datasets to contain the same count and order of features.

## Monitoring Progress
Since training is often an iterative process, it is useful to obtain real-time feedback as to how the learner is progressing. For example, you may want to monitor the training loss to make sure that it isn't increasing instead of decreasing with training. Such early feedback can indicate model overfitting or improperly tuned hyper-parameters. Learners that implement the [Verbose](verbose.md) interface accept a [PSR-3](https://www.php-fig.org/psr/psr-3/) logger instance that can be used to output training information at each time step (or *epoch*). The library comes built-in with a [Screen Logger](other/loggers/screen.md) that does the job for most cases.

```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('example'));

$estimator->train($dataset);
```

```sh
[2020-02-08 06:02:57] example.INFO: Learner init booster=RegressionTree(max_depth=4 max_leaf_size=3 max_features=null min_purity_increase=1.0E-7) rate=0.1 ratio=0.5 estimators=1000 min_change=
0.0001 window=10 hold_out=0.1 metric=RSquared base=DummyRegressor(strategy=Mean)
[2020-02-08 06:02:57] example.INFO: Training base learner
[2020-02-08 06:02:58] example.INFO: Epoch 1 score=0.097751119226963 loss=6281265049.8042
[2020-02-08 06:02:58] example.INFO: Epoch 2 score=0.20067635817301 loss=5537137575.1759
[2020-02-08 06:02:58] example.INFO: Epoch 3 score=0.28331060230153 loss=4869582841.9896
[2020-02-08 06:02:59] example.INFO: Epoch 4 score=0.36730710929531 loss=4370856640.0286
[2020-02-08 06:02:59] example.INFO: Epoch 5 score=0.44183724381919 loss=3869890119.1739
...
[2020-02-08 06:04:07] example.INFO: Epoch 189 score=0.91416877793484 loss=192224708.33229
[2020-02-08 06:04:07] example.INFO: Epoch 190 score=0.91412704964864 loss=191766700.22592
[2020-02-08 06:04:08] example.INFO: Epoch 191 score=0.91395637122531 loss=191053995.23096
[2020-02-08 06:04:08] example.INFO: Epoch 192 score=0.91420102466788 loss=189569206.71289
[2020-02-08 06:04:08] example.INFO: Epoch 193 score=0.91410009100939 loss=188312560.73359
[2020-02-08 06:04:08] example.INFO: Ensemble restored to epoch 183
[2020-02-08 06:04:08] example.INFO: Training complete
```

## Feature Importances
Learners that implement the [Ranks Features](ranks-features.md) interface can determine the importance of each feature in the training set. Feature importances are defined as the degree to which an individual feature influences the outcome of the prediction. Feature importances are useful for feature selection and for explaining predictions derived from a model. To output the normalized importance scores, call the `featureImportances()` method of a trained learner that implements the Ranks Features interface.

```php
// Train learner

$importances = $estimator->featureImportances();

var_dump($importances);
```

```sh
array(4) {
  [0]=> float(0.047576266783176)
  [1]=> float(0.3794817175945)
  [2]=> float(0.53170249909942)
  [3]=> float(0.041239516522901)
}
```