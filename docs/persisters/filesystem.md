<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Persisters/Filesystem.php">[source]</a></span>

# Filesystem
Filesystems are local or remote storage drives that are organized by files and folders. The Filesystem persister saves models to a file at a given path and can automatically keep a history of past saved models.

This class uses the local filesystem by default, but remote filesystems (such as Amazon S3, Azure Blob Storage, Dropbox, PDO Database etc...) are also supported by configuring the `Filesystem` object and passing it to the constructor of this class. 

See: [Filesystem Adapters](https://github.com/thephpleague/flysystem#adapters).

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | path | | `string` | The path to the model file on the filesystem. |
| 2 | history | `false` | `bool` | Should we keep a history of past saves? |
| 3 | serializer | `Native` | `Serializer` | The serializer used to convert to and from storage format. |
| 4 | filesystem | `Storage::local()` | `Filesystem` | The filesystem to use as a storage backend. By default the local filesystem is used |

## Example
```php
use Rubix\ML\Other\Helpers\Storage;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Persisters\Serializers\Igbinary;

$persister = new Filesystem('/path/to/example.model', true, new Iginary(), Storage::local());
```

## Additional Methods
This persister does not have any additional methods.
