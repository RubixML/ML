<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/PersitentModel.php">[source]</a></span>

# Persistent Model
The Persistent Model meta-estimator wraps a [Persistable](persistable.md) learner with additional functionality for saving and loading the model. It uses [Persister](persisters/api.md) objects to interface with various storage backends such as the [Filesystem](persisters/filesystem.md) or [Redis](persisters/redis-db.md).

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Probabilistic](probabilistic.md), [Wrapper](wrapper.md)

**Data Type Compatibility:** Depends on base learner

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | Learner | An instance of a persistable estimator. |
| 2 | persister | | Persister | The persister object used interface with the storage medium. |

## Additional Methods
Save the persistent model to storage:
```php
public save() : void
```

Load the persistent model from storage given a persister:
```php
public static load(Persister $persister) : self
```

## Example
```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('example.model'));

// Do something ...

$estimator->save();
```