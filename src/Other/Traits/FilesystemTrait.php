<?php

namespace Rubix\ML\Other\Traits;

use League\Flysystem\FilesystemInterface;
use Rubix\ML\Other\Helpers\Storage;

/**
 * Filesystem Aware
 *
 * This trait can be used in classes that internally read or write to a filesystem. It provides a default implementation
 * of \Rubix\ML\FilesystemAware interface
 *
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait FilesystemTrait
{
    /**
     * The Filesystem instance to perform filesystem operations using
     *
     * @var ?FilesystemInterface
     */
    protected $filesystem;

    public function setFilesystem(FilesystemInterface $filesystem) : void
    {
        $this->filesystem = $filesystem;
    }

    /**
     * Return the target filesystem instance.
     * If the user has not provided one when instantiating, or not invoked setFilesystem()  then this method
     * returns a reference to to local filesystem.
     *
     * @return FilesystemInterface
     */
    public function filesystem() : FilesystemInterface
    {
        if (!$this->filesystem) {
            $this->filesystem = Storage::local();
        }

        return $this->filesystem;
    }
}
