<?php

namespace Rubix\ML\Storage;

/**
 * Writer
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 */
interface Writer
{
    /**
     * Write.
     *
     * @param string $location
     * @param mixed $data
     * @throws \Rubix\ML\Storage\Exceptions\WriteError
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     */
    public function write(string $location, $data) : void;

    /**
     * Move.
     *
     * NOTE: If supported by the underlying datastore this should be implemented as an atomic operation.
     *
     * @param string $from
     * @param string $to
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     */
    public function move(string $from, string $to) : void;

    /**
     * Delete.
     *
     * @param string $location
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     */
    public function delete(string $location) : void;
}
