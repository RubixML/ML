<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Filesystem.php">[source]</a></span>

# Filesystem
Filesystems are local or remote storage drives that are organized by files and folders. The Filesystem persister saves models to a file at a given path and can automatically keep a history of past saved models.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | path | | string | The path to the model file on the filesystem. |
| 2 | history | false | bool | Should we keep a history of past saves? |
| 3 | serializer | Native | Serializer | The serializer used to convert to and from storage format. |

## Example
```php
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Persisters\Serializers\RBX;

$persister = new Filesystem('/path/to/example.model', true, new RBX());
```

## Additional Methods
This persister does not have any additional methods.
