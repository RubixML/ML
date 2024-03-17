<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Backends/Swoole.php">[source]</a></span>

# Swoole
[Swoole](https://swoole.com)/[OpenSwoole](https://openswoole.com/) is an async PHP extension that also supports multiprocessing.

!!! tip
    Swoole backend makes use of [Igbinary](https://www.php.net/manual/en/intro.igbinary.php) serializer. If you need to
    optimize the memory usage (or getting out-of-memory errors) consider installing [Igbinary](https://www.php.net/manual/en/intro.igbinary.php).

## Example

No parameters are required. It's a drop-in replacement for the [Serial](backends/serial.md) backend.

```php
use Rubix\ML\Backends\Swoole;

$backend = new Swoole();
```