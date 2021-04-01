# Persisters
Persisters are responsible for persisting Encoding objects to storage and are also used by the [Persistent Model](../persistent-model.md) meta-estimator to save and restore models that have been serialized.

### Save
To save an encoding:
```php
public save(Encoding $encoding) : void
```

```php
$persister->save($encoding);
```

### Load
To load an encoding from persistence:
```php
public load() : Encoding
```

```php
$encoding = $persister->load();
```