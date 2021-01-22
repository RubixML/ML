<?php

namespace Rubix\ML\Other\Traits;

use function hash;
use function get_object_vars;
use function implode;
use function sort;

/**
 * Tracks Revisions
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait TracksRevisions
{
    /**
     * Return the revision number of the class.
     *
     * @return string
     */
    public function revision() : string
    {
        $properties = array_keys(get_object_vars($this));

        sort($properties);

        return hash('crc32b', implode(':', $properties));
    }
}
