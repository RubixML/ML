# Training
Most estimators have the ability to be trained with data. Estimators that require training are called [Learners](learner.md) and implement the `train()` method among others. Training is the process of feeding data to a learner so that it can build an internal representation (or *model*) of the problem. Supervised learners require a dataset with labels that act as a signal to guide the learner. Unsupervised learners can be trained with either a labeled or unlabeled dataset but only the samples are used to build the model.

To begin training a learner, pass a dataset object to the `train()` method on the learner instance like in the example below.

```php
$estimator->train($dataset);
```

## Batch vs Online Learning
Batch learning is when a learner is trained in full using only one dataset in a single session. Calling the `train()` method on a learner instance is an example of batch learning. In contrast, *online* learning occurs when a learner is trained over multiple sessions with multiple datasets as small as a single sample each. Learners that are capable of being partially trained like this implement the [Online](online.md) interface which includes the `partial()` method for training in an online scheme. Subsequent calls to the `partial()` method will continue training where the learner left off. Online learning is especially useful for when you have a dataset that is too large to fit into memory all at once.

```php
$folds = $dataset->fold(3);

$estimator->train($folds[0]);

$estimator->partial($folds[1]);

$estimator->partial($folds[2]);
```

> **Note:** After the initial training, the learner will expect subsequent datasets to contain the same number and order of features.

## Monitoring Progress
Since training is often an iterative process, it is useful to obtain feedback as to how the learner is progressing in real-time. For example, you may want to monitor the training loss to make sure that it isn't increasing instead of decreasing with training. Such early feedback can indicate model overfitting or improperly tuned hyper-parameters. Learners that implement the [Verbose](verbose.md) interface accept a [PSR-3](https://www.php-fig.org/psr/psr-3/) logger instance that can be used to output training information at each time step (or *epoch*). The library comes built-in with a [Screen Logger](other/loggers/screen.md) that does the job for most cases.

```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen());

$estimator->train($dataset);
```

```sh
[2020-07-12 23:52:59] housing.INFO: Learner init Gradient Boost {booster: Regression Tree {max_height: 4, max_leaf_size: 3, max_features: null, min_purity_increase: 1.0E-7}, rate: 0.1, ratio: 0
.5, estimators: 1000, min_change: 0.0001, window: 10, hold_out: 0.1, metric: RMSE, base: Dummy Regressor {strategy: Mean}}
[2020-07-12 23:52:59] INFO: Training started
[2020-07-12 23:52:59] INFO: Training base learner
[2020-07-12 23:52:59] INFO: Epoch 1 - RMSE: -75966.144307681, L2 Loss: 6273028418.4053
[2020-07-12 23:52:59] INFO: Epoch 2 - RMSE: -71837.423145166, L2 Loss: 5398183359.0029
[2020-07-12 23:52:59] INFO: Epoch 3 - RMSE: -67949.979096606, L2 Loss: 4847398522.703
[2020-07-12 23:53:00] INFO: Epoch 4 - RMSE: -63802.65363341, L2 Loss: 4515203001.2578
[2020-07-12 23:53:00] INFO: Epoch 5 - RMSE: -61624.027074156, L2 Loss: 3988666807.5813
...
[2020-07-12 23:53:06] INFO: Epoch 67 - RMSE: -24464.486466965, L2 Loss: 706106123.97902
[2020-07-12 23:53:06] INFO: Epoch 68 - RMSE: -24473.10530312, L2 Loss: 701284659.63732
[2020-07-12 23:53:06] INFO: Epoch 69 - RMSE: -24347.871068021, L2 Loss: 696422563.57693
[2020-07-12 23:53:06] INFO: Epoch 70 - RMSE: -24328.676819944, L2 Loss: 690861140.1853
[2020-07-12 23:53:06] INFO: Epoch 71 - RMSE: -24068.607669273, L2 Loss: 685669903.74276
[2020-07-12 23:53:06] INFO: Restoring ensemble state to epoch 61
[2020-07-12 23:53:06] INFO: Training complete
```

## Feature Importances
Learners that implement the [Ranks Features](ranks-features.md) interface can evaluate the importance of each feature in the training set. Feature importances are defined as the degree to which an individual feature influences the model. Feature importances are useful for feature selection and for helping to explain predictions derived from a model. To output the normalized importance scores, call the `featureImportances()` method of a trained learner that implements the Ranks Features interface.

```php
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new ClassificationTree(10);

$estimator->train($dataset);

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
