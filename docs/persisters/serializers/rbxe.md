<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/RBXE.php">[source]</a></span>

# RBX Encrypted
Encrypted Rubix Object File format (RBXE) is a format to securely store and share serialized PHP objects. In addition to providing verifiability like the standard [RBX](./rbxp.md) format, RBXE encrypts the file data so that it cannot be read without the password.

> **Note:** Requires the PHP [Open SSL extension](https://www.php.net/manual/en/book.openssl.php) to be installed.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | password | '' | string | The secret key used to sign and verify HMACs. |
| 2 | base | Gzip | Serializer | The base serializer. The recommended default is Gzipped native PHP serialization format. |

## Example
```php
use Rubix\ML\Persisters\Serializers\RBXE;
use Rubix\ML\Persisters\Serializers\Gzip;
use Rubix\ML\Persisters\Serializers\Native;

$serializer = new RBXE('secret', new Gzip(9, new Native()));
```

### References
>- H. Krawczyk et al. (1997). HMAC: Keyed-Hashing for Message Authentication.
