<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/RBXE.php">[source]</a></span>

# RBX Encrypted
Encrypted Rubix Object File format (RBXE) is a format to securely store and share serialized PHP objects. In addition to providing verifiability like the standard [RBX](./rbx.md) format, RBXE encrypts the file data so that it cannot be read without the password. 

> **Note:** Requires the PHP [Open SSL extension](https://www.php.net/manual/en/book.openssl.php) to be installed.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | password | '' | string | The secret key used to encrypt the data. |
| 2 | base | Native | Serializer | The base serializer. |

## Example
```php
use Rubix\ML\Persisters\Serializers\RBXE;
use Rubix\ML\Persisters\Serializers\Native;

$serializer = new RBXE('secret', new Native());
```

### References
>- H. Krawczyk et al. (1997). HMAC: Keyed-Hashing for Message Authentication.
