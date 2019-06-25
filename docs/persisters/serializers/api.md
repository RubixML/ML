### Serializers
Serializers take Persistable estimators and convert them between their in-memory and storage representations.

To serialize a Persistable:
```php
public serialize(Persistable $persistable) : string
```

To unserialize a Persistable:
```php
public unserialize(string $data) : Persistable
```