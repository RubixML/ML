<?php

namespace Rubix\ML;

use League\Flysystem\FilesystemInterface;
use Rubix\ML\Other\Helpers\Storage;
use RuntimeException;
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
     * @param FilesystemInterface|null $filesystem The filesystem to use. If null this method uses the local filesystem
     * @throws \Exception If the file cannot be successfully written to the filesystem
     */
    public function write(string $path, ?FilesystemInterface $filesystem = null) : void
    {
        if (!$filesystem) {
            $filesystem = Storage::local();
        }

        $success = $filesystem->write($path, $this->data());

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
