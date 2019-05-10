<?php

namespace Rubix\ML\Backends;

use Closure;

interface Backend
{
    /**
     * Queue up a function for backend processing.
     *
     * @param callable $function
     * @param array $args
     * @param \Closure|null $after
     */
    public function enqueue(callable $function, array $args = [], ?Closure $after = null) : void;

    /**
     * Process the queue and return the results.
     *
     * @return array
     */
    public function process() : array;

    /**
     * Flush the queue.
     */
    public function flush() : void;
}
