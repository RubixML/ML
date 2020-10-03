<?php

namespace Rubix\ML\Storage;

use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Storage\Streams\Stream;

/**
 * Wraps a Datastore (that implements both Reader and Writer) to be used as a Reader.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 */
class ReadProxy implements Reader
{
    /**
     * @var \Rubix\ML\Storage\Reader
     */
    protected $storage;

    /**
     * @param \Rubix\ML\Storage\Reader $storage
     */
    public function __construct(Reader $storage)
    {
        $this->storage = $storage;
    }

    /**
     * Return if the target exists at $location.
     *
     * @param string $location
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     * @return bool
     */
    public function exists(string $location) : bool
    {
        return $this->storage->exists($location);
    }

    /**
     * Open a stream of the target data at $location.
     *
     * @param string $location
     * @param string $mode
     * @throws \Rubix\ML\Storage\Exceptions\ReadError
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     * @return \Rubix\ML\Storage\Streams\Stream
     */
    public function read(string $location, string $mode = Stream::READ_ONLY) : Stream
    {
        return $this->storage->read($location, $mode);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Read Proxy (storage: ' . Params::toString($this->storage) . ')';
    }
}
