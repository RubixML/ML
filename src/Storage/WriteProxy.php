<?php

namespace Rubix\ML\Storage;

use Rubix\ML\Other\Helpers\Params;

/**
 * Wraps a Datastore (that implements both Reader and Writer) to be used as a Writer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 */
class WriteProxy implements Writer
{
    /**
     * @var \Rubix\ML\Storage\Writer
     */
    protected $storage;

    public function __construct(Writer $storage)
    {
        $this->storage = $storage;
    }

    /**
     * @param string $location
     * @param mixed $data
     * @throws \Rubix\ML\Storage\Exceptions\WriteError
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     */
    public function write(string $location, $data) : void
    {
        $this->storage->write($location, $data);
    }

    /**
     * @param string $from
     * @param string $to
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     */
    public function move(string $from, string $to) : void
    {
        $this->storage->move($from, $to);
    }

    /**
     * Delete.
     *
     * @param string $location
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     */
    public function delete(string $location) : void
    {
        $this->storage->delete($location);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Write Proxy (storage: ' . Params::toString($this->storage) . ')';
    }
}
