### Persisters
Persisters are responsible for persisting a *Persistable* object to storage and are also used by the [Persistent Model](#persistent-model) meta-estimator to save and restore models.

To store a persistable estimator:
```php
public save(Persistable $persistable) : void
```

Load the saved model from persistence:
```php
public load() : Persistable
```