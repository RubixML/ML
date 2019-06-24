### Online
Certain estimators that implement the *Online* interface can be trained in batches. Estimators of this type are great for when you either have a continuous stream of data or a dataset that is too large to fit into memory. Partial training allows the model to evolve as new information about the world is acquired.

> **Note**: Learner will continue to train as long as you are using the `partial()` method, however, calling `train()` on a trained or partially trained learner will reset it back to baseline first.

To partially train an online learner:
```php
public partial(Dataset $dataset) : void
```

**Example:**
```php
$folds = $dataset->fold(3);

$estimator->train($folds[0]);

$estimator->partial($folds[1]);

$estimator->partial($folds[2]);
```