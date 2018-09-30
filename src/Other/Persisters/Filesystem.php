<?php

namespace Rubix\ML\Other\Persisters;

use Rubix\ML\Persistable;
use InvalidArgumentException;
use RuntimeException;

class Filesystem implements Persister
{
    /**
     * The path to the file on the filesystem.
     *
     * @var string
     */
    protected $path;

    /**
     * Should we overwrite an already existing file?
     *
     * @var bool
     */
    protected $overwrite;

    /**
     * @param  string  $path
     * @param  bool  $overwrite
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $path, bool $overwrite = false)
    {
        if (!is_writable(dirname($path))) {
            throw new InvalidArgumentException('Folder does not exist or is not'
                . ' writable. Check path and permissions.');
        }

        $this->path = $path;
        $this->overwrite = $overwrite;
    }

    /**
     * Return an associative array of file info or false if file does not exist.
     *
     * @return array|null
     */
    public function info() : ?array
    {
        $info = stat($this->path);

        if ($info === false) {
            return null;
        }

        return array_slice($info, 13);
    }

    /**
     * Restore the persistable object.
     *
     * @throws \RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function restore() : Persistable
    {
        if (!file_exists($this->path)) {
            throw new RuntimeException('File ' . basename($this->path)
                . ' cannot be opened. Check the path.');
        }

        if (!is_readable($this->path)) {
            throw new RuntimeException('File ' . basename($this->path)
                . ' cannot be opened. Check the file permissions.');
        }

        $persistable = unserialize(file_get_contents($this->path) ?: '');

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Object cannot be reconstituted.');
        }

        return $persistable;
    }

    /**
     * Save the persitable object.
     *
     * @param  \Rubix\ML\Persistable  $persistable
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return void
     */
    public function save(Persistable $persistable) : void
    {
        if ($this->overwrite === false and file_exists($this->path)) {
            throw new RuntimeException('Attempting to overwrite an existing'
                . ' model.');
        }

        $success = file_put_contents($this->path, serialize($persistable), LOCK_EX);

        if (!$success) {
            throw new RuntimeException('Failed to serialize object to a file.');
        }
    }
}
