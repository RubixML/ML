<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/Gzip.php">[source]</a></span>

# Gzip
A compression format based on the DEFLATE algorithm with a header and CRC32 checksum, the Gzip serializer outputs smaller model files at the cost of a bit more processing during serialization and deserialization. In addition, the Gzip serializer can be applied to any other non-compressed serialization format by changing the base serializer.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | level | 6 | int | The compression level between 0 and 9, 0 meaning no compression. |

## Example
```php
use Rubix\ML\Serializers\Gzip;

$serializer = new Gzip(1);
```

## References
[^1]: P. Deutsch. (1996). RFC 1951 - DEFLATE Compressed Data Format Specification version.
