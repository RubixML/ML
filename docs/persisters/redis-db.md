<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Persisters/RedisDB.php">Source</a></span>

# Redis DB
Redis is a high performance in-memory key value store that can be used to persist your trained models over a network.

> **Note:** The Redis persister requires the PHP [Redis extension](https://github.com/phpredis/phpredis) and a properly configured Redis server.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | key | | string | The key of the object in the database. |
| 2 | host | '127.0.0.1' | string | The hostname or IP address of the Redis server. |
| 3 | port | 6379 | int | The port of the Redis server. |
| 4 | db | 0 | int | The database number. |
| 5 | password | None | string | An optional password to access a password-protected server. |
| 6 | serializer | Native | object | The serializer used to convert to and from storage format. |
| 7 | timeout | 2.5 | float | The time in seconds to wait for a response from the server before timing out. |

### Additional Methods
Return an associative array of info from the Redis server:
```php
public info() : array
```

### Example
```php
use Rubix\ML\Persisters\RedisDB;
use Rubix\ML\Persisters\Serializers\Native;

$persister = new RedisDB('model:sentiment', '127.0.0.1', 6379, 2, 'secret', new Native(), 2.5);
```