### Learner
Most estimators have the ability to be trained with data. These estimators are called *Learners* and require training before they are able make predictions. Training is the process of feeding data to the learner so that it can formulate a generalized function that maps future samples to good predictions.

> **Note**: Calling `train()` on an already trained estimator will cause any previous training to be lost. If you would like to be able to train a model incrementally, see the [Online](#online) Estimator interface.

To train an learner pass it a training dataset:
```php
public train(Dataset $training) : void
```

Return whether or not the learner has been trained:
```php
public trained() : bool
```

**Example:**

```php
$estimator->train($dataset);

var_dump($estimator->trained());
```

**Output:**

```sh
bool(true)
```