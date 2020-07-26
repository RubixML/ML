# Ranks Features
The Ranks Features interface is for learners that can determine the importances of the features used to train them. Low importance is given to feature columns that do not contribute significantly in the model whereas high importance indicates that the feature is more influential. Feature importances can help explain the predictions derived from a model and can also be used to identify informative features for feature selection.

### Feature Importances
Return the normalized importance scores of each feature column of the training set:
```php
public featureImportances() : array
```

```php
$estimator->train($dataset);

$importances = $estimator->featureImportances();

var_dump($importances);
```

```sh
array(4) {
  [0]=> float(0.047576266783176)
  [1]=> float(0.3794817175945)
  [2]=> float(0.53170249909942)
  [3]=> float(0.041239516522901)
}
```
