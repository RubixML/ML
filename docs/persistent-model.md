<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/PersistentModel.php">[source]</a></span>

# Persistent Model
The Persistent Model meta-estimator wraps a [Persistable](persistable.md) learner with additional functionality for saving and loading the model. It uses [Persister](persisters/api.md) objects to interface with various storage backends such as the [Filesystem](persisters/filesystem.md) or [Redis](persisters/redis-db.md).

**Interfaces:** [Wrapper](wrapper.md), [Estimator](estimator.md), [Learner](learner.md), [Probabilistic](probabilistic.md), [Scoring](scoring.md)

**Data Type Compatibility:** Depends on base learner

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | Persistable | The persistable base learner. |
| 2 | persister | | Persister | The persister used to interface with the storage medium. |

## Examples
```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(new KMeans(10), new Filesystem('example.model'));
```

## Additional Methods
Load the model:
```php
public static load(Persister $persister) : self
```

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('example.model'));
```

Save the model:
```php
public save() : void
```

```php
$estimator->save();
```
