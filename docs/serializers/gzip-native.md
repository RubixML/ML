<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/GzipNative.php">[source]</a></span>

# Gzip Native
Gzip Native wraps the native PHP serialization format in an outer compression layer based on the DEFLATE algorithm with a header and CRC32 checksum.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | level | 6 | int | The compression level between 0 and 9, 0 meaning no compression. |

## Example
```php
use Rubix\ML\Serializers\GzipNative;

$serializer = new GzipNative(1);
```

## References
[^1]: P. Deutsch. (1996). RFC 1951 - DEFLATE Compressed Data Format Specification version.
