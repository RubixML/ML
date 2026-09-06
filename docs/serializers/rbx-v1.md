<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/RBXV1.php">[source]</a></span>

# RBX V1

Rubix Object File format (RBX) is a format designed to reliably store and share serialized PHP objects. Based on PHP's native serialization format, RBX adds additional layers of compression, data integrity checks, and class compatibility detection all in one robust format.

!!! warning
    This is the legacy RBX v1 format, which was gzip-based. It has been superseded by [RBX V2](rbxv2.md), which is the default serializer.

!!! note
    We recommend to use the `.rbx` file extension when storing RBX-serialized PHP objects.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | level | 6 | int | The compression level between 0 and 9, 0 meaning no compression. |

## Example

```php
use Rubix\ML\Serializers\RBXV1;

$serializer = new RBXV1(6);
```
