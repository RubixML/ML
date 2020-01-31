<?php

namespace Rubix\ML\Backends;

use Rubix\ML\Deferred;

interface Backend
{
    /**
     * Queue up a deferred computation for backend processing.
     *
     * @param \Rubix\ML\Deferred $computation
     * @param callable|null $after
     */
    public function enqueue(Deferred $computation, ?callable $after = null) : void;

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
}
