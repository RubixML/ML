<?php

namespace Rubix\ML;

use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Storage\Exceptions\StorageException;
use Rubix\ML\Storage\LocalFilesystem;
use Rubix\ML\Storage\WriteProxy;
use Rubix\ML\Storage\Writer;
use Stringable;

class Encoding implements Stringable
{
    /**
     * The encoded data.
     *
     * @var string
     */
    protected $data;

    /**
     * @param string $data
     */
    public function __construct(string $data)
    {
        $this->data = $data;
    }

    /**
     * Return the encoded data.
     *
     * @return string
     */
    public function data() : string
    {
        return $this->data;
    }

    /**
     * Return the size of the encoding in bytes.
     *
     * @return int
     */
    public function bytes() : int
    {
        return strlen($this->data);
    }

    /**
     * Write the encoding to a file at the path specified.
     *
     * @param string $path
     * @param ?Writer $storage
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function write(string $path, ?Writer $storage = null) : void
    {
        if (!$storage) {
            $storage = new LocalFilesystem();
        }

        $storage = new WriteProxy($storage);

        try {
            $storage->write($path, $this->data);
        } catch (StorageException $e) {
            throw new RuntimeException($e->getMessage(), $e->getCode(), $e);
        }
    }

    /**
     * Return the object as a string.
     *
     * @return string
     */
    public function __toString() : string
    {
        return $this->data;
    }
}
