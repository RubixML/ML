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
     * @param \Rubix\ML\Backends\Tasks\Task $task
     * @param callable(mixed):void|null $after
     */
    public function enqueue(Task $task, ?callable $after = null) : void;

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
}
