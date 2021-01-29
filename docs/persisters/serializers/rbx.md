<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/RBX.php">[source]</a></span>

# RBX
Rubix Object File format (RBX) is a format designed to reliably store and share serialized PHP objects. Based on PHP's native serialization format, RBX adds additional layers of compression, data integrity checks, and class compatibility detection all in one robust format.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | Gzip | Serializer | The base serializer. |

## Example
```php
use Rubix\ML\Persisters\Serializers\RBX;
use Rubix\ML\Persisters\Serializers\Gzip;

$serializer = new RBX(new Gzip(1));
```
