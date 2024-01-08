<?php

namespace Rubix\ML\Backends\Swoole;

use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Tasks\Task;
use RuntimeException;
use Swoole\Coroutine\Scheduler;

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
        /**
         * Swoole promises that all the coroutines added to the root of the
         * scheduler will be executed in parallel.
         */
        $scheduler = new Scheduler();

        $results = [];

        foreach ($this->queue as $callback) {
            $scheduler->add(function () use ($callback, &$results) {
                $results[] = $callback();
            });
        }

        if (!$scheduler->start()) {
            throw new RuntimeException('Not all coroutines finished successfully');
        }

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
