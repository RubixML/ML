<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/PersistentModel.php">[source]</a></span>

# Persistent Model
The Persistent Model meta-estimator wraps a [Persistable](persistable.md) learner with additional functionality for saving and loading the model. It uses [Persister](persisters/api.md) objects to interface with various storage backends such as the [Filesystem](persisters/filesystem.md).

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Probabilistic](probabilistic.md), [Scoring](scoring.md)

**Data Type Compatibility:** Depends on base learner

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | Persistable | The persistable base learner. |
| 2 | persister | | Persister | The persister used to interface with the storage system. |
| 3 | serializer | RBX | Serializer | The object serializer. |

## Examples
```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Serializers\RBX;

$estimator = new PersistentModel(new KMeans(10), new Filesystem('example.model'), new RBX());
```

## Additional Methods
Load the model from storage:
```php
public static load(Persister $persister, ?Serializer $serializer = null) : self
```

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Serializers\RBX;

$estimator = PersistentModel::load(new Filesystem('example.model'), new RBX());
```

Save the model to storage:
```php
public save() : void
```

```php
$estimator->save();
```
