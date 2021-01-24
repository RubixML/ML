<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/RBXP.php">[source]</a></span>

# RBXP
Rubix Object File format (RBXP) is a format designed to securely and reliably store and share serialized PHP objects. Based on PHP's native serialization format, RBXP adds additional layers of compression, tamper protection, and class compatibility detection all in one robust format.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | password | '' | string | The secret key used to sign and verify HMACs. |
| 2 | base | Gzip | Serializer | The base serializer. Default is Gzipped native PHP serialization format. |

## Example
```php
use Rubix\ML\Persisters\Serializers\RBXP;

$serializer = new RBXP('secret', new Gzip(1, new Native()));
```
