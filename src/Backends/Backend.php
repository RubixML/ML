<?php

namespace Rubix\ML\Backends;

use Rubix\ML\Backends\Tasks\Task;

interface Backend
{
    /**
     * Queue up a task for backend processing.
     *
     * @param \Rubix\ML\Backends\Tasks\Task $task
     * @param callable(mixed):void|null $after
     */
    public function enqueue(Task $task, ?callable $after = null) : void;

    /**
     * Process the queue and return the results.
     *
     * @return mixed[]
     */
    public function process() : array;

    /**
     * Flush the queue.
     */
    public function flush() : void;

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
