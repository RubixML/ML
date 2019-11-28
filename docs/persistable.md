# Persistable
If an estimator implements the Persistable interface then its model data can be saved and loaded by a [Persister](persisters/api.md) or by wrapping it with a [Persistent Model](persistent-model.md) meta estimator. This interface provides no additional methods otherwise.

**Example**

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Regressors\Adaline;

$persistable = new Adaline(200);

$estimator = new PersistentModel($persistable, new Filesystem('example.model'));

$estimator->save();
```