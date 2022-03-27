<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/RBX.php">[source]</a></span>

# RBX
Rubix Object File format (RBX) is a format designed to reliably store and share serialized PHP objects. Based on PHP's native serialization format, RBX adds additional layers of compression, data integrity checks, and class compatibility detection all in one robust format.

!!! note
    We recommend to use the `.rbx` file extension when storing RBX-serialized PHP objects.

## Parameters
## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | level | 6 | int | The compression level between 0 and 9, 0 meaning no compression. |

## Example
```php
use Rubix\ML\Serializers\RBX;

$serializer = new RBX(6);
```
