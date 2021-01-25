# Online
Learners that implement the Online interface can be trained in batches. Learners of this type are great for when you either have a continuous stream of data or a dataset that is too large to fit into memory. In addition, partial training allows the model to evolve over time.

## Partially Train
To partially train an Online learner pass it a training set to its `partial()` method:
```php
public partial(Dataset $dataset) : void
```

```php
$folds = $dataset->fold(3);

$estimator->partial($folds[0]);

$estimator->partial($folds[1]);

$estimator->partial($folds[2]);
```

!!! note
    Learner will continue to train as long as you are using the `partial()` method, however, calling `train()` on a trained or partially trained learner will reset it back to baseline first.