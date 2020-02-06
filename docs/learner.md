# Learner
Most estimators have the ability to be trained with data. These estimators are called *Learners* and require training before they are can make predictions. Training is the process of feeding data to the learner so that it can form a generalized internal function that maps future unknown samples to good predictions.

### Train a Learner
To train a learner pass a training dataset to the `train()` method:
```php
public train(Dataset $training) : void
```

**Example**

```php
$estimator->train($dataset);
```

> **Note:** Calling the `train()` method on an already trained learner will erase its previous training. If you would like to train a model incrementally, you can do so with learners implementing the [Online](online.md) interface.

### Is the Learner Trained?
Return whether or not the learner has been trained:
```php
public trained() : bool
```

**Example**

```php
var_dump($estimator->trained());
```

```sh
bool(true)
```

### Predict a Single Sample
Pass a single sample through the model and return the prediction:
```php
public predictSample(array $sample) : mixed
```

**Example**

```php
$prediction = $estimator->predictSample([4, 'furry', 'loner', 8.65]);

var_dump($prediction);
```

```sh
string(3) "cat"
```