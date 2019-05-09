<?php

namespace Rubix\ML\Backends;

/**
 * Serial
 *
 * The Serial backend executes tasks sequentially inside of a single PHP process.
 * The advantage of the Serial backend is that it has zero overhead due to
 * parallelization, thus it may be faster than a parallel backend in cases where
 * the computions are minimal such as with small datasets.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Serial implements Backend
{
    /**
     * An array of tuples containing a callable and its args.
     *
     * @var array[]
     */
    protected $queue = [
        //
    ];

    /**
     * Queue up a function for backend processing.
     *
     * @param callable $function
     * @param array $args
     * @param callable $after
     */
    public function enqueue(callable $function, array $args = [], ?callable $after = null) : void
    {
        $this->queue[] = [$function, $args, $after];
    }

    /**
     * Process the queue.
     *
     * @return array
     */
    public function process() : array
    {
        $results = [];

        foreach ($this->queue as [$function, $args, $after]) {
            $result = $function(...$args);

            if ($after) {
                $after($result);
            }

            $results[] = $result;
        }

        $this->queue = [];

        return $results;
    }
}
