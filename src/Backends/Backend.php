<?php

namespace Rubix\ML\Backends;

use Rubix\ML\Backends\Tasks\Task;
use Stringable;

/**
 * Backend
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Backend extends Stringable
{
    /**
     * Queue up a task for backend processing.
     *
     * @internal
     *
     * @param Task $task
     * @param ?callable $after
     */
    public function enqueue(Task $task, ?callable $after = null) : void;

    /**
     * Return the number of concurrent worker processes.
     *
     * @internal
     *
     * @return int
     */
    public function workers() : int;

    /**
     * Process the queue and return the results.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function process() : array;

    /**
     * Flush the queue.
     *
     * @internal
     */
    public function flush() : void;

    /**
     * Gracefully shut down the backend and release any resources.
     *
     * @internal
     */
    public function shutdown() : void;
}
