# Ranks Features
The Ranks Features interface is for learners that can determine the importances of the features used to train them. Low importance is given to feature columns that do not contribute significantly in the model whereas high importance indicates that the feature is more influential. Feature importances can help explain the predictions derived from a model and can also be used to identify informative features for feature selection.

### Feature Importances
Return the importance scores of each feature column of the training set:
```php
public featureImportances() : array
```

```php
$estimator->train($dataset);

$importances = $estimator->featureImportances();

print_r($importances);
```

```php
Array
(
    [0] => 0.04757
    [1] => 0.37948
    [2] => 0.53170
    [3] => 0.04123
)
```
