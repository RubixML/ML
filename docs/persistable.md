# Persistable
An estimator that implements the Persistable interface can be serialized by a [Serializer](serializers/api.md) or save and loaded using the [Persistent Model](persistent-model.md) meta-estimator.

To return the current class revision hash:
```php
public revision() : string
```

```php
echo $persistable->revision();
```

```
e7eeec9a
```
