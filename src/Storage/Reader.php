<?php

namespace Rubix\ML\Storage;

use Rubix\ML\Storage\Streams\Stream;

/**
 * Reader
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 */
interface Reader
{
    /**
     * Return if the target exists at $location.
     *
     * @param string $location
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     * @return bool
     */
    public function exists(string $location) : bool;

    /**
     * Open a stream of the target data at $location.
     *
     * @param string $location
     * @param string $mode
     * @throws \Rubix\ML\Storage\Exceptions\ReadError
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     * @return \Rubix\ML\Storage\Streams\Stream
     */
    public function read(string $location, string $mode = Stream::READ_ONLY) : Stream;
}
