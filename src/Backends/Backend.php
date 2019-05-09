<?php

namespace Rubix\ML\Backends;

interface Backend
{
    /**
     * Queue up a function for backend processing.
     *
     * @param callable $function
     * @param array $args
     * @param callable|null $after
     */
    public function enqueue(callable $function, array $args = [], ?callable $after = null) : void;

    /**
     * Process the queue.
     *
     * @return array
     */
    public function process() : array;
}
