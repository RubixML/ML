<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Serializers/RBXV2.php">[source]</a></span>

# RBX V2

Rubix Object File format v2 (RBX) is an improvement upon the RBX V1 format that adds additional layers of security and integrity checks to ensure that serialized objects are not tampered with or corrupted during storage or transmission.

RBX V2 is the default serializer used by the [Persistent Model](../persistent-model.md) meta-estimator. It replaces the legacy gzip-based [RBX V1](rbx-v1.md) format.

!!! note
    We recommend to use the `.rbx` file extension when storing RBX-serialized PHP objects.

## Parameters

RBXV2 does not have any constructor parameters.

## Example

```php
use Rubix\ML\Serializers\RBXV2;

$serializer = new RBXV2();
```
