# Serializers
Serializers take objects that implement the [Persistable](../persistable.md) interface and convert them into blobs of data called *encodings*. Encodings can then be used to either store an object or to reinstantiate an object from storage.

### Serialize
To serialize a persistable object into an encoding:
```php
public serialize(Persistable $persistable) : Encoding
```

```php
$encoding = $serializer->serialize($persistable);
```

### Deserialize
To deserialize a persistable object from an encoding:
```php
public deserialize(Encoding $encoding) : Persistable
```

```php
$persistable = $serializer->deserialize($encoding);
```
