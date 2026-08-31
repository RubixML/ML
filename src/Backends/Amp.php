<?php

namespace Rubix\ML\Backends;

use Rubix\ML\Helpers\CPU;
use Rubix\ML\Backends\Tasks\Task;
use Amp\Parallel\Worker\ContextWorkerPool;
use Amp\Parallel\Worker\LimitedWorkerPool;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Amp
 *
 * Amp Parallel is a multiprocessing subsystem that requires no extensions. It uses a
 * non-blocking concurrency framework based on fibers and the Revolt event loop.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Amp implements Backend
{
    /**
     * The worker pool.
     *
     * @var LimitedWorkerPool
     */
    protected LimitedWorkerPool $pool;

    /**
     * A 3-tuple of executions and their optional callbacks and contexts.
     *
     * @var list<array{\Amp\Parallel\Worker\Execution<mixed,mixed,mixed>,callable(mixed,mixed):void|null,mixed|null}>
     */
    protected array $queue = [
        //
    ];

    /**
     * @param int|null $workers
     * @throws InvalidArgumentException
     */
    public function __construct(?int $workers = null)
    {
        if (isset($workers) and $workers < 1) {
            throw new InvalidArgumentException('Number of workers'
                . " must be greater than 0, $workers given.");
        }

        $this->pool = new ContextWorkerPool($workers ?? CPU::cores());
    }

    /**
     * Return the number of background worker processes.
     *
     * @return int
     */
    public function workers() : int
    {
        return $this->pool->getWorkerLimit();
    }

    /**
     * Queue up a deferred task for backend processing.
     *
     * @internal
     *
     * @param Task $task
     * @param callable(mixed,mixed):void $after
     * @param mixed $context
     */
    public function enqueue(Task $task, ?callable $after = null, mixed $context = null) : void
    {
        $execution = $this->pool->submit($task);

        $this->queue[] = [$execution, $after, $context];
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
        $results = [];

        foreach ($this->queue as [$execution, $after, $context]) {
            $result = $execution->await();

            if ($after) {
                $after($result, $context);
            }

            $results[] = $result;
        }

        $this->flush();

        return $results;
    }

    /**
     * Flush the queue and clear the memorized results.
     *
     * @internal
     */
    public function flush() : void
    {
        $this->queue = [];
    }

    /**
     * Gracefully shut down the worker pool.
     *
     * @internal
     */
    public function shutdown() : void
    {
        $this->pool->shutdown();
    }

    /**
     * @return array{workers: int}
     */
    public function __serialize() : array
    {
        return ['workers' => $this->workers()];
    }

    /**
     * @param array{workers: int} $data
     */
    public function __unserialize(array $data) : void
    {
        $this->pool = new ContextWorkerPool($data['workers']);
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
        return "Amp (workers: {$this->workers()})";
    }
}
