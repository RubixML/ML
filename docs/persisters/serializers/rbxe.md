<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/RBXE.php">[source]</a></span>

# RBX Encrypted
Encrypted Rubix Object File format (RBXE) is a format to securely store and share serialized PHP objects. In addition to ensuring data integrity like RBX format, RBXE also adds layers of security such as tamper protection and data encryption while being resilient to brute-force and evasive to timing attacks.

!!! note
    Requires the PHP [Open SSL extension](https://www.php.net/manual/en/book.openssl.php) to be installed.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | password | '' | string | The password used to sign and encrypt the data. |
| 2 | base | Native | Serializer | The base serializer. |

## Example
```php
use Rubix\ML\Persisters\Serializers\RBXE;
use Rubix\ML\Persisters\Serializers\Native;

$serializer = new RBXE('secret', new Native());
```

### References
>- H. Krawczyk et al. (1997). HMAC: Keyed-Hashing for Message Authentication.
