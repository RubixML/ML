<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Persisters/Filesystem.php">[source]</a></span>

# Filesystem
Filesystems are local or remote storage drives that are organized by files and folders. The Filesystem persister saves models to a file at a given path and can automatically keep a history of past saved models.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | path | | string | The path to the model file on the filesystem. |
| 2 | history | false | bool | Should we keep a history of past saves? |

## Example
```php
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Serializers\RBX;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;

$persistable = new KNearestNeighbors(3, false, new Manhattan());

$persister = new Filesystem('/path/to/example.rbx', true);

$serializer = new RBX(6);

$encoding = $serializer->serialize($persistable);

$persister->save($encoding);
```

## Example
```php
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Serializers\RBX;

$persister = new Filesystem('/path/to/example.rbx', true);

$encoding = $persister->load();

$serializer = new RBX(6);

$persistable = $serializer->deserialize($encoding);
```

## Additional Methods
This persister does not have any additional methods.
