<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CommitteeMachine.php">Source</a></span>

# Committee Machine
A voting ensemble that aggregates the predictions of a committee of heterogeneous learners (referred to as *experts*). The committee employs a user-specified influence-based scheme to make final predictions.

> **Note:** Influence values can be arbitrary as they are normalized upon instantiation.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Parallel](parallel.md), [Verbose](verbose.md), [Persistable](persistable.md)

**Data Type Compatibility:** Depends on the base learners

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | experts | | array | An array of learner instances that will comprise the committee. |
| 2 | influences | Equal | array | The influence score for each expert in the committee. |

### Additional Methods
Return the learner instances of the committee:
```php
public experts() : array
```

Return the normalized influence scores of each expert in the committee:
```php
public influences() : array
```

### Example
```php
use Rubix\ML\CommitteeMachine;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;

$estimator = new CommitteeMachine([
    new GaussianNB(),
    new RandomForest(new ClassificationTree(4), 100, 0.3),
    new KDNeighbors(3),
    new SoftmaxClassifier(100, new Mometum(0.001)),
], [
    1, 4, 3, 2,
]);
```

### References
>- [1] H. Drucker. (1997). Fast Committee Machines for Regression and Classification.