<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Serializers/RBXV2.php">[source]</a></span>

# RBXV2

Rubix Object File format v2 (RBX) is a format designed to reliably store and share serialized PHP objects. It is built directly on PHP's native serialization format and layers on top of it data-integrity checksums, class-compatibility detection, and a hardened deserialization path that restricts which classes are permitted to be reconstructed, all in one compact format.

RBXV2 is the default serializer used by the [Persistent Model](../persistent-model.md) meta-estimator. It replaces the legacy gzip-based [RBX](rbx.md) format, which is no longer read by this serializer.

!!! note
    We recommend to use the `.rbx` file extension when storing RBX-serialized PHP objects.

## Parameters

RBXV2 does not have any constructor parameters.

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |

## Example

```php
use Rubix\ML\Serializers\RBXV2;

$serializer = new RBXV2();
```
