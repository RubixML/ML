<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persisters\Serializers\Serializer;

use Rubix\ML\Storage\LocalFilesystem;

/**
 * Filesystem
 *
 * Filesystems are local or remote storage drives that are organized by files
 * and folders. The filesystem persister serializes models to a file at a
 * user-specified path.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Chris Simpson
 */
class Filesystem extends HistoryPersister implements Persister
{
    /**
     * Creates a `Persister` backed by `\Rubix\ML\Storage\LocalFilesystem`
     *
     * @param string $location
     * @param bool $history
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $serializer
     */
    public function __construct(string $location, bool $history = false, ?Serializer $serializer = null)
    {
        parent::__construct($location, new LocalFilesystem(), $history, $serializer);
    }
}
