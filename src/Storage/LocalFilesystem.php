<?php

namespace Rubix\ML\Storage;

use Rubix\ML\Storage\Streams\File;
use Rubix\ML\Storage\Streams\Stream;
use Rubix\ML\Storage\Exceptions\RuntimeException;

/**
 * Local Datastore.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 */
class LocalFilesystem implements Datastore
{
    /**
     * @see \Rubix\ML\Storage\Reader::exists()
     * @inheritdoc
     *
     * @param string $location
     *
     * @return bool
     */
    public function exists(string $location) : bool
    {
        return file_exists($location);
    }

    /**
     * @see \Rubix\ML\Storage\Reader::read()
     * @inheritdoc
     *
     * @param string $location
     *
     * @return \Rubix\ML\Storage\Streams\Stream
     */
    public function read(string $location, string $mode = Stream::READ_ONLY) : Stream
    {
        $resource = fopen($location, $mode);

        if (!$resource) {
            throw new RuntimeException("Could not open $location.");
        }

        $stream = new File($resource);

        if (!$stream->readable()) {
            throw new RuntimeException("Stream with mode {$mode} cannot be read from");
        }

        return $stream;
    }

    /**
     * @see \Rubix\ML\Storage\Writer::write()
     * @inheritdoc
     *
     * @param string $location
     * @param \Rubix\ML\Storage\Streams\Stream|string $data
     */
    public function write(string $location, $data) : void
    {
        if ($data instanceof Stream) {
            $data = $data->contents();
        }

        file_put_contents($location, $data, LOCK_EX);
    }

    /**
     * @see \Rubix\ML\Storage\Writer::delete()
     * @inheritdoc
     *
     * @param string $location
     */
    public function delete(string $location) : void
    {
        unlink($location);
    }

    /**
     * @see \Rubix\ML\Storage\Writer::move()
     * @inheritdoc
     *
     * @param string $from
     * @param string $to
     */
    public function move(string $from, string $to) : void
    {
        rename($from, $to);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Local Filesystem';
    }
}
