<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/RBX.php">[source]</a></span>

# RBX
Rubix Object File format (RBX) is a format designed to reliably store and share serialized PHP objects. Based on PHP's native serialization format, RBX adds additional layers of compression, data integrity checks, and class compatibility detection all in one robust format.

!!! note
    We recommend to use the extension `.rbx` when storing RBX-formatted object files.

## Parameters
## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | Gzip | Gzip | The base Gzip serializer. |

## Example
```php
use Rubix\ML\Serializers\RBX;
use Rubix\ML\Serializers\Gzip;

$serializer = new RBX(new Gzip(9));
```
