<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Flysystem.php">[source]</a></span>

# Flysystem
[Flysystem](https://flysystem.thephpleague.com) is a filesystem library providing a unified storage interface and abstraction layer. It enables access to many different storage backends such as Local, Amazon S3, FTP, and more.

!!! note
    The Flysystem persister is designed to work with Flysystem version 2.0.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | path | | string | The path to the persistable object file on the filesystem. |
| 2 | filesystem |  | FilesystemOperator | The Flysystem filesystem operator responsible for read and write operations. |
| 3 | history | false | bool | Should we keep a history of past saves? |
| 4 | serializer | Native | Serializer | The serializer used to convert to and from storage format. |

## Example
```php
use League\Flysystem\Filesystem;
use League\Flysystem\Local\LocalFilesystemAdapter;
use Rubix\ML\Persisters\Flysystem;
use Rubix\ML\Persisters\Serializers\Native;

$filesystem = new Filesystem(new LocalFilesystemAdapter('/path/to/'));

$persister = new Flysystem('example.model', $filesystem, true, new Native());
```

## Additional Methods
This persister does not have any additional methods.
