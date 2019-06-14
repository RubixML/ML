<?php

namespace Rubix\ML\Backends;

use Rubix\ML\Deferred;
use Closure;

/**
 * Serial
 *
 * The Serial backend executes tasks sequentially inside of a single PHP process.
 * The advantage of the Serial backend is that it has zero overhead, thus it may
 * be faster than a parallel backend in cases where the computions are minimal
 * such as with small datasets.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Serial implements Backend
{
    /**
     * A 2-tuple of deferred computations and their callbacks.
     *
     * @var array[]
     */
    protected $queue = [
        //
    ];

    /**
     * Return the number of background worker processes.
     *
     * @return int
     */
    public function workers() : int
    {
        return 1;
    }

    /**
     * Queue up a deferred computation for backend processing.
     *
     * @param \Rubix\ML\Deferred $deferred
     * @param \Closure|null $after
     */
    public function enqueue(Deferred $deferred, ?Closure $after = null) : void
    {
        $this->queue[] = [$deferred, $after];
    }

    /**
     * Process the queue and return the results.
     *
     * @return array
     */
    public function process() : array
    {
        $results = [];

        foreach ($this->queue as [$deferred, $after]) {
            $result = $deferred->compute();

            if ($after) {
                $after($result);
            }

            $results[] = $result;
        }

        $this->queue = [];

        return $results;
    }

    /**
     * Flush the queue.
     */
    public function flush() : void
    {
        $this->queue = [];
    }
}
