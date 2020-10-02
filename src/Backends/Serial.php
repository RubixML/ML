<?php

namespace Rubix\ML\Backends;

use Rubix\ML\Backends\Tasks\Task;

/**
 * Serial
 *
 * The Serial backend executes tasks sequentially inside of a single PHP process. The
 * advantage of the Serial backend is that it has zero overhead, thus it may be faster
 * than a parallel backend in cases where the computations are minimal such as with
 * small datasets.
 *
 * > **Note:** The Serial backend is the default for most objects that are capable of
 * parallel processing.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Serial implements Backend
{
    /**
     * A 2-tuple of deferred computations and their optional callbacks.
     *
     * @var array[]
     */
    protected $queue = [
        //
    ];

    /**
     * Queue up a deferred computation for backend processing.
     *
     * @param \Rubix\ML\Backends\Tasks\Task $task
     * @param callable(mixed):void|null $after
     */
    public function enqueue(Task $task, ?callable $after = null) : void
    {
        $this->queue[] = [$task, $after];
    }

    /**
     * Process the queue and return the results.
     *
     * @return mixed[]
     */
    public function process() : array
    {
        $results = [];

        foreach ($this->queue as [$task, $after]) {
            $result = $task();

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

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Serial';
    }
}
