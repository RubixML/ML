<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/PersitentModel.php">[source]</a></span>

# Persistent Model
The Persistent Model wrapper gives the estimator two additional methods (`save()` and `load()`) that allow a [Persistable](persistable.md) learner to be saved and retrieved from storage. Persistent Model uses [Persister](./persisters/api.md) objects under the hood that allow the model to be stored in different locations.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Probabilistic](probabilistic.md)

**Data Type Compatibility:** Depends on base learner

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | object | An instance of the base estimator to be persisted. |
| 2 | persister | | object | The persister object used to store the model data. |

### Additional Methods
Save the persistent model to storage:
```php
public save() : void
```

Load the persistent model from storage given a persister:
```php
public static load(Persister $persister) : self
```

### Example
```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Persisters\Serializers\Native;

$persister = new Filesystem('random_forest.model', true, new Native());

$estimator = new PersistentModel(new RandomForest(100), $persister);
```