<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/ClassificationTree.php">[source]</a></span>

# Classification Tree
A binary tree-based learner that greedily constructs a decision map for classification that minimizes the Gini impurity among the training labels within the leaf nodes. The height and *bushiness* of the tree can be determined by the user-defined `max height` and `max leaf size` hyper-parameters. Classification Trees also serve as the base learner of ensemble methods such as [Random Forest](random-forest.md) and [AdaBoost](adaboost.md).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | maxHeight | PHP_INT_MAX | int | The maximum height of the tree. |
| 2 | maxLeafSize | 3 | int | The max number of samples that a leaf node can contain. |
| 3 | maxFeatures | Auto | int | The max number of feature columns to consider when determining a best split. |
| 4 | minPurityIncrease | 1e-7 | float | The minimum increase in purity necessary for a node *not* to be post pruned during tree growth. |

## Example
```php
use Rubix\ML\Classifiers\ClassificationTree;

$estimator = new ClassificationTree(10, 7, 4, 0.01);
```

## Additional Methods
Return a human-readable text representation of the decision tree ruleset:
```php
public rules(?array $header = null) : string
```

```php
echo $estimator->rules(['age', 'height', 'income']);
```

```sh
├─── age < 70
├───├─── income < 260734.0
├───├───├─── income < 80207.0
├───├───├───├─── height < 182.0
├───├───├───├───├─── Best (outcome=high school impurity=0.19546677755182 n=9)
├───├───├───├─── height >= 182.0
├───├───├───├───├─── Best (outcome=bachelors impurity=-0 n=67)
├───├───├─── income >= 80207.0
├───├───├───├─── Best (outcome=masters impurity=-0 n=77)
├───├─── income >= 260.73460601
├───├───├─── Best (outcome=doctorate impurity=-0 n=49)
├─── age >= 70
├───├─── Best (outcome=high school impurity=-0 n=98)
```

Return the height of the tree i.e. the number of layers:
```php
public height() : int
```

Return the balance factor of the tree:
```php
public balance() : int
```

## References:
[^1]: W. Y. Loh. (2011). Classification and Regression Trees.
[^2]: K. Alsabti. et al. (1998). CLOUDS: A Decision Tree Classifier for Large Datasets.