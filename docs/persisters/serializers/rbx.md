<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/RBX.php">[source]</a></span>

# RBX
Rubix Object File format (RBX) is a format designed to reliably store and share serialized PHP objects. Based on PHP's native serialization format, RBX adds additional layers of compression, tamper protection, and class compatibility detection all in one robust format. Unlike the encrypted [RBXE](./rbxe.md) however, file data can still be read even if the authenticity of it cannot be verified with the password.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | password | '' | string | The secret key used to sign HMACs and encrypt the data. |
| 2 | base | Gzip | Serializer | The base serializer. Default is Gzipped native PHP serialization format. |

## Example
```php
use Rubix\ML\Persisters\Serializers\RBX;
use Rubix\ML\Persisters\Serializers\Gzip;
use Rubix\ML\Persisters\Serializers\Native;

$serializer = new RBX('secret', new Gzip(1, new Native()));
```

### References
>- H. Krawczyk et al. (1997). HMAC: Keyed-Hashing for Message Authentication.
