<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CommitteeMachine.php">[source]</a></span>

# Committee Machine
A voting ensemble that aggregates the predictions of a committee of heterogeneous learners (referred to as *experts*). The committee employs a user-specified influence scheme to weight the final predictions.

!!! note
    Influence values can be on any arbitrary scale as they are automatically normalized upon instantiation.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Parallel](parallel.md), [Persistable](persistable.md)

**Data Type Compatibility:** Depends on the base learners

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | experts | | array | An array of learner instances that will comprise the committee. |
| 2 | influences | null | array | The influence values for each expert in the committee. If null, each expert will be weighted equally. |

## Example
```php
use Rubix\ML\CommitteeMachine;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\Classifiers\SoftmaxClassifier;

$estimator = new CommitteeMachine([
    new GaussianNB(),
    new RandomForest(new ClassificationTree(4), 100, 0.3),
    new KDNeighbors(3),
    new SoftmaxClassifier(100),
], [
    0.2, 0.4, 0.3, 0.1,
]);
```

## Additional Methods
Return the learner instances of the committee:
```php
public experts() : array
```

Return the normalized influence scores of each expert in the committee:
```php
public influences() : array
```

## References
[^1]: H. Drucker. (1997). Fast Committee Machines for Regression and Classification.