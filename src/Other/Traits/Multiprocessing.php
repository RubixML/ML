<?php

namespace Rubix\ML\Other\Traits;

use InvalidArgumentException;

use const Rubix\ML\DEFAULT_WORKERS;

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
     * The max number of processes to run in parallel.
     *
     * @var int
     */
    protected $workers = DEFAULT_WORKERS;

    /**
     * Set the maximum number of processes to run in parallel.
     *
     * @param int $workers
     * @throws \InvalidArgumentException
     */
    public function setWorkers(int $workers) : void
    {
        if ($workers < 1) {
            throw new InvalidArgumentException('Cannot run less than'
                . " 1 worker process, $workers given.");
        }

        $this->workers = $workers;
    }

    /**
     * Return the maximum number of workers.
     *
     * @return int
     */
    public function workers() : int
    {
        return $this->workers;
    }
}
