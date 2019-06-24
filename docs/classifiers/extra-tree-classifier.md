<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/ExtraTreeClassifier.php">Source</a></span></p>

# Extra Tree Classifier
An *Extremely Randomized* Classification Tree - these trees differ from standard [Classification Trees](#classification-tree) in that they choose the best split drawn from a random set determined by *max features*, rather than searching the entire column. Extra Trees work well in ensembles such as [Random Forest](#random-forest) or [AdaBoost](#adaboost) as the *weak learner* or they can be used on their own. The strength of Extra Trees are computational efficiency as well as increasing variance of the prediction (if that is desired).

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Data Type Compatibility:** Categorical, Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max depth | PHP_INT_MAX | int | The maximum depth of a branch. |
| 2 | max leaf size | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | min purity increase | 0. | float | The minimum increase in purity necessary for a node *not* to be post pruned. |
| 4 | max features | Auto | int | The max number of features to consider when determining a best split. |
| 5 | tolerance | 1e-3 | float | A small amount of impurity to tolerate when choosing a best split. |

### Additional Methods
Return the feature importances calculated during training indexed by feature column:
```php
public featureImportances() : array
```

Return the height of the tree:
```php
public height() : int
```

Return the balance of the tree:
```php
public balance() : int
```

### Example
```php
use Rubix\ML\Classifiers\ExtraTreeClassifier;

$estimator = new ExtraTreeClassifier(50, 3, 0.10, 4, 1e-3);
```

### References
>- P. Geurts et al. (2005). Extremely Randomized Trees.