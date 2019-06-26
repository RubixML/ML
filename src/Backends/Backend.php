<?php

namespace Rubix\ML\Backends;

use Rubix\ML\Deferred;
use Closure;

interface Backend
{
    /**
     * Queue up a deferred computation for backend processing.
     *
     * @param \Rubix\ML\Deferred $computation
     * @param \Closure|null $after
     */
    public function enqueue(Deferred $computation, ?Closure $after = null) : void;

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
