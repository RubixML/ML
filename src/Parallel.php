<?php

namespace Rubix\ML;

use Rubix\ML\Backends\Backend;

/**
 * Parallel
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Parallel
{
    /**
     * Set the parallel processing backend.
     *
     * @param Backend $backend
     */
    public function setBackend(Backend $backend) : void;
}
