<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Persisters/Flysystem.php">[source]</a></span>

# Flysystem
[Flysystem](https://flysystem.thephpleague.com) is a filesystem abstraction library providing a unified interface for 
[many different filesystems](https://github.com/thephpleague/flysystem#adapters). It enables access to remote storage backends such as Amazon S3, Azure Blob Storage, Google Cloud Storage, Dropbox...

The Flysystem persister saves models to a file at a given path using the Flysystem library, and can automatically keep a history of past saved models.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | path | | `string` | The path to the model file on the filesystem. |
| 2 | filesystem |  | `FilesystemInterface` | The flysystem object providing access to your storage backend |
| 3 | serializer | `Native` | `Serializer` | The serializer used to convert to and from storage format. |

## Examples

### Local:
 Using the Flysystem Persister to interact with data stored on your local filesystem:
- Pass the Flysystem Persister a `Filesystem` object that uses the `Local` adapter instance:
```php
use League\Flysystem\Filesystem;
use League\Flysystem\Adapter\Local;
use Rubix\ML\Persisters\Flysystem;

$storage = new Filesystem(new Local('/'));
$persister = new Flysystem('/path/to/example.model', $storage);

// Or, to save keystrokes you could also use the shortcut method:

$persister = Flysystem::local('/path/to/example.model');
```

### FTP Server:
Using the Flysystem Persister to interact with data stored on an FTP Server:
- Pass the Flysystem Persister a `Filesystem` object that uses the `Ftp` adapter instance:

```php
use League\Flysystem\Filesystem;
use League\Flysystem\Adapter\Ftp;
use Rubix\ML\Persisters\Flysystem;

$config = [
    'host' => 'ftp.example.com',
    'username' => 'username',
    'password' => 'password',
];
$storage = new Filesystem(new Ftp($config));
$persister = new Flysystem('/path/to/example.model', $storage);

// Or, to save keystrokes you could also use the shortcut method:

$persister = Flysystem::ftp('/path/to/example.model', $config);
```

### Amazon S3
Using the Flysystem Persister to interact with data on Amazon S3:
- install the Flysystem S3 adapter: `composer require league/flysystem-aws-s3-v3`
- Pass the Flysystem Persister a `Filesystem` object that uses the `AwsS3Adapter` adapter instance:

```php
use Aws\S3\S3Client;
use League\Flysystem\Filesystem;
use League\Flysystem\AwsS3v3\AwsS3Adapter;
use Rubix\ML\Persisters\Flysystem;

$client = new S3Client([
    'credentials' => [
        'key'    => 'your-key',
        'secret' => 'your-secret',
    ],
    'region' => 'your-region',
    'version' => 'latest|version',
]);

$storage = new Filesystem(new AwsS3Adapter($client, 'your-bucket-name'));

$persister = new Flysystem('/path/to/example.model', $storage);
```


## Additional Methods

Shortcut to return a Flysystem Persister backed by the [Local](https://flysystem.thephpleague.com/v1/docs/adapter/local/) filesystem

```php
public static local(string $path, bool $history = false, ?Serializer $serializer = null) : self
```

Shortcut to return a Flysystem Persister backed by an [FTP Server](https://flysystem.thephpleague.com/v1/docs/adapter/ftp/).

```php
public static ftp(string $path, array $config, bool $history = false, ?Serializer $serializer = null) : self
```
