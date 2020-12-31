<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Persisters\Serializers\Serializer;
use League\Flysystem\Filesystem as FlyFilesystem;
use League\Flysystem\Local\LocalFilesystemAdapter;

use function dirname;
use function basename;

/**
 * Filesystem
 *
 * Filesystems are local storage drives that are organized by files and folders. The filesystem
 * persister serializes persistable objects to a file at a user-specified path.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Filesystem extends Flysystem
{
    /**
     * @param string $path
     * @param bool $history
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $serializer
     */
    public function __construct(string $path, bool $history = false, ?Serializer $serializer = null)
    {
        $filesystem = new FlyFilesystem(new LocalFilesystemAdapter(dirname($path)));

        parent::__construct(basename($path), $filesystem, $history, $serializer);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Filesystem (path: {$this->path},"
            . ' history: ' . Params::toString($this->history) . ','
            . " serializer: {$this->serializer})";
    }
}
