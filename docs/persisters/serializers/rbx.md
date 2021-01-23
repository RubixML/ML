<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/RBX.php">[source]</a></span>

# RBX
Rubix Object File format (RBX) is a format designed to securely and reliably store and share serialized PHP objects. Based on PHP's native serialization format, RBX adds additional layers of compression, tamper protection, and class compatibility detection all in one robust format.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | password | '' | string | The secret key used to sign and verify HMACs. |
| 2 | compressionLevel | 9 | int | The compression level between 0 and 9, 0 meaning no compression. |

## Example
```php
use Rubix\ML\Persisters\Serializers\RBX;

$serializer = new RBX(1);
```
