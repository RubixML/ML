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
     * A 3-tuple of deferred computations and their optional callbacks and contexts.
     *
     * @var list<array{Task,callable(mixed,mixed):void|null,mixed|null}>
     */
    protected array $queue = [
        //
    ];

    /**
     * Queue up a deferred computation for backend processing.
     *
     * @param Task $task
     * @param callable(mixed,mixed):void|null $after
     * @param mixed|null $context
     */
    public function enqueue(Task $task, ?callable $after = null, $context = null) : void
    {
        $this->queue[] = [$task, $after, $context];
    }

    /**
     * Process the queue and return the results.
     *
     * @return mixed[]
     */
    public function process() : array
    {
        $results = [];

        foreach ($this->queue as [$task, $after, $context]) {
            $result = $task();

            if ($after) {
                $after($result, $context);
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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Serial';
    }
}
