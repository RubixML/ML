<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/RandomForest.php">Source</a></span>

# Random Forest
Ensemble classifier that trains Decision Trees ([Classification Trees](classification-tree.md) or [Extra Trees](extra-tree-classifier.md)) on a random subset (*bootstrap* set) of the training data. A prediction is made based on the probability scores returned from each tree in the forest averaged and weighted equally.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Parallel](../parallel.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | ClassificationTree | object | The base tree estimator. The default is a fully grown Classification Tree. |
| 2 | estimators | 100 | int | The number of estimators to train in the ensemble. |
| 3 | ratio | 0.2 | float | The ratio of random samples (between 0 and 1.5) to train each estimator with. |

### Additional Methods
This estimator does not have any additional methods.

### Example
```php
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new RandomForest(new ClassificationTree(10), 300, 0.1);
```

### References
>- L. Breiman. (2001). Random Forests.
>- L. Breiman et al. (2005). Extremely Randomized Trees.