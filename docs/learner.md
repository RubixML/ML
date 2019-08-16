# Learner
Most estimators have the ability to be trained with data. These estimators are called *Learners* and require training before they are can make predictions. Training is the process of feeding data to the learner so that it can form a generalized internal function that maps future unknown samples to good predictions.

### Train a Learner
To train a learner pass it a training dataset:
```php
public train(Dataset $training) : void
```

**Example**

```php
$estimator->train($dataset);
```

> **Note:** Calling the `train()` method on an already trained learner will erase the previous training. If you would like to train a model incrementally, refer to the [Online](online.md) interface.

### Is the Learner Trained?
Return whether or not the learner has been trained:
```php
public trained() : bool
```

**Example**

```php
var_dump($estimator->trained());
```

**Output**

```sh
bool(true)
```

### Predict Single Sample
Predict a single sample and return the result:
```php
public predictSample(array $sample) : mixed
```

**Example**

```php
// Import samples

$dataset = new Unlabeled($samples);

$prediction = $estimator->predictSample($dataset[2]); // Predict the third sample in dataset

var_dump($prediction);
```

**Output**

```sh
string(3) "cat"
```