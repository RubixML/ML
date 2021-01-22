<?php

namespace Rubix\ML\Other\Traits;

use function sha1;
use function get_object_vars;

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
     * The revision number of the class.
     *
     * @return string
     */
    public function revision() : string
    {
        $properties = array_keys(get_object_vars($this));

        sort($properties);

        return sha1(implode(':', $properties));
    }
}
