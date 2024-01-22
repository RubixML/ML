<?php

namespace Rubix\ML\Traits;

use Rubix\ML\Backends\Backend;

/**
 * Multiprocessing
 *
 * Multiprocessing is the use of two or more processes that usually execute on
 * multiple cores when training. Estimators that implement the Parallel interface
 * can take advantage of multiple core systems by executing parts or all of the
 * algorithm in parallel.
 *
 * > **Note**: The optimal number of workers will depend on the system
 * specifications of the computer. Fewer workers than CPU cores can result in
 * slower performance but too many workers can cause excess overhead.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait Multiprocessing
{
    /**
     * The parallel processing backend.
     *
     * @var Backend
     */
    protected Backend $backend;

    /**
     * Set the parallel processing backend.
     *
     * @param Backend $backend
     */
    public function setBackend(Backend $backend) : void
    {
        $this->backend = $backend;
    }
}
