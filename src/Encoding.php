<?php

namespace Rubix\ML;

use Rubix\ML\Exceptions\RuntimeException;
use Stringable;

use function strlen;
use function dirname;
use function is_file;
use function file_put_contents;

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
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function write(string $path) : void
    {
        if (!is_file($path) and !is_writable(dirname($path))) {
            throw new RuntimeException('Folder does not exist or is not writable');
        }

        if (is_file($path) and !is_writable($path)) {
            throw new RuntimeException("File $path is not writable.");
        }

        $success = file_put_contents($path, $this->data, LOCK_EX);

        if (!$success) {
            throw new RuntimeException('Failed to write to the filesystem.');
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
