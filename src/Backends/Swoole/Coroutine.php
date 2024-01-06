<?php

namespace Rubix\ML\Backends\Swoole;

use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Tasks\Task;

use function Swoole\Coroutine\batch;

/**
 * Swoole
 *
 * Works both with Swoole and OpenSwoole.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 */
class Coroutine implements Backend
{
    /**
     * The queue of tasks to be processed in parallel.
     *
     * @var list<callable>
     */
    protected array $queue = [
        //
    ];

    protected float $timeout;

    public function __construct(float $timeout = -1)
    {
        $this->timeout = $timeout;
    }

    /**
     * Queue up a deferred task for backend processing.
     *
     * @internal
     *
     * @param \Rubix\ML\Backends\Tasks\Task $task
     * @param callable(mixed,mixed):void $after
     * @param mixed $context
     */
    public function enqueue(Task $task, ?callable $after = null, $context = null) : void
    {
        $this->queue[] = function () use ($task, $after, $context) {
            $result = $task();

            if ($after) {
                $after($result, $context);
            }

            return $result;
        };
    }

    /**
     * Process the queue and return the results.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function process() : array
    {
        $results = batch($this->queue, $this->timeout);

        $this->queue = [];

        return $results;
    }

    /**
     * Flush the queue
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
        return 'Swoole\\Coroutine';
    }
}
