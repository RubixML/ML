<?php

namespace Rubix\Engine\Persisters;

use InvalidArgumentException;
use RuntimeException;

class Filesystem implements Persister
{
    const MODE = 'wb+';

    /**
     * The path to the file in the filesystem.
     *
     * @var string
     */
    protected $path;

    /**
     * @param  string  $path
     * @return void
     */
    public function __construct(string $path)
    {
        if (!is_writable(dirname($path))) {
            throw new InvalidArgumentException('Folder does not exist or is not writeable.');
        }

        $this->path = $path;
    }

    /**
     * Save the model to the Filesystem. Returns true on success, false on error.
     *
     * @return bool
     */
    public function save(Persistable $model) : bool
    {
        return file_put_contents($this->path, serialize($model), LOCK_EX) ? true : false;
    }

    /**
     * Restore the estimator from the filesystem.
     *
     * @throws \RuntimeException
     * @return \Rubix\Engine\Persistable
     */
    public function restore() : Persistable
    {
        if (!file_exists($this->path) || !is_readable($this->path)) {
            throw new RuntimeException('File ' . basename($path) . ' cannot be opened.');
        }

        $model = unserialize(file_get_contents($this->path));

        if ($model === false) {
            throw new RuntimeException('Model could not be reconstituted.');
        }

        return $model;
    }
}
