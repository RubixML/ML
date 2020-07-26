# Persisters
Persisters are responsible for persisting a [Persistable](../persistable.md) object to storage and are also used by the [Persistent Model](../persistent-model.md) meta-estimator to save and restore models.

### Save
To store a persistable object:
```php
public save(Persistable $persistable) : void
```

```php
$persister->save($estimator);
```

### Load
Load the saved object from persistence:
```php
public load() : Persistable
```

```php
$estimator = $persister->load();
```