<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Serializers/RBX.php">[source]</a></span>

# RBX
Rubix Object File Format (RBX) is a format designed to reliably store serialized PHP objects. Based on PHP's native serialization format, RBX includes additional features such as compression, tamper protection, and class definition compatibility detection.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | level | 9 | int | The compression level between 0 and 9, 0 meaning no compression. |

## Example
```php
use Rubix\ML\Persisters\Serializers\RBX;

$serializer = new RBX(9);
```

## Specification
The following is the specification for the RBX file format.

### File Structure
RBX files are partitioned into four sections with newline character (`\n`) delimiters demarcating each section.

#### Identification String
The first section contains the identification string or *magic number* which identifies the RBX format contained in the file.

#### Checksum
The next section contains a CRC32B hash of the contents of the file header. Upon deserialization, this checksum must be used to verify the contents of the header before trusting the header data.

#### Header
The header section contains a JSON object of metadata related to the file's contents. The following headers must be included with each file.

1. An integer `version` property.
2. A `class` object containing the class `name` of the object data and its `revision` number.
3. A `data` object that contains the data `format`, `compression`, `checksum`, and `length` properties.

#### Body
The last section contains the object data in the format specified by the header. The data must be validated by the CRC32B checksum found in the header or else an error should be thrown.
