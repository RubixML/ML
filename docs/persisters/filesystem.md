<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Persisters/Filesystem.php">Source</a></span>

# Filesystem
Filesystems are local or remote storage drives that are organized by files and folders. The Filesystem persister saves models to a file at a given path and automatically keeps backups of the latest versions of your models.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | path | | string | The path to the model file on the filesystem. |
| 2 | history | false | bool | Should we keep a history of past saves? |
| 3 | serializer | Native | object | The serializer used to convert to and from storage format. |

### Additional Methods
This persister does not have any additional methods.

### Example
```php
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Persisters\Serializers\Binary;

$persister = new Filesystem('/path/to/example.model', true, new Binary());
```