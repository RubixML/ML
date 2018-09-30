<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persistable;
use InvalidArgumentException;
use RuntimeException;

/**
 * Filesystem
 *
 * Filesystems are local or remote storage drives that are organized by files
 * and folders. The filesystem persister serializes models to a file at a
 * user-specified path.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
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
    public function __construct(string $path, bool $overwrite = true)
    {
        if (!is_writable(dirname($path))) {
            throw new InvalidArgumentException('Folder does not exist or is not'
                . ' writable. Check path and permissions.');
        }

        $this->path = $path;
        $this->overwrite = $overwrite;
    }

    /**
     * Return the size of the file on the filesystem in bytes.
     *
     * @return int|null
     */
    public function size() : ?int
    {
        return filesize($this->path) ?: null;
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
                . ' does not exist. Check the path.');
        }

        $data = file_get_contents($this->path) ?: '';

        $persistable = unserialize($data);

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
            throw new RuntimeException('Cannot overwrite existing file.');
        }

        $data = serialize($persistable);

        $success = file_put_contents($this->path, $data, LOCK_EX);

        if (!$success) {
            throw new RuntimeException('Failed to serialize object to'
                . ' filesystem.');
        }
    }

    /**
     * Remove the file from the filesystem.
     *
     * @return void
     */
    public function delete() : void
    {
        unlink($this->path);
    }
}
