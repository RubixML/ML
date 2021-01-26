# Learner
Most estimators have the ability to be trained with data. These estimators are called *Learners* and require training before they are can make predictions. Training is the process of feeding data to the learner so that it can form a generalized representation or *model* of the dataset.

### Train a Learner
To train a learner pass a training dataset as argument to the `train()` method:
```php
public train(Dataset $training) : void
```

```php
$estimator->train($dataset);
```

!!! note
    Calling the `train()` method on an already trained learner will erase its previous training. If you would like to train a model incrementally, you can do so with learners implementing the [Online](online.md) interface.

### Is the Learner Trained?
Return whether or not the learner has been trained:
```php
public trained() : bool
```

```php
var_dump($estimator->trained());
```

```sh
bool(true)
```
