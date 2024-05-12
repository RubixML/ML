<?php

namespace Rubix\ML\Backends;

use Amp\Future;
use Amp\Parallel\Worker\ContextWorkerPool;
use Amp\Parallel\Worker\Task as AmpTask;
use Rubix\ML\Backends\Amp\CallableTask;
use Rubix\ML\Backends\Tasks\Task;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Helpers\CPU;
use function Amp\async;

/**
 * Amp
 *
 * Amp Parallel is a multiprocessing subsystem that requires no extensions. It uses a
 * non-blocking concurrency framework that implements coroutines using PHP generator
 * functions under the hood.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Amp2 implements Backend
{
    /**
     * The worker pool.
     *
     * @var ContextWorkerPool
     */
    protected ContextWorkerPool $pool;

    /**
     * The queue of coroutines to be processed in parallel.
     *
     * @var Future<mixed>[]
     */
    protected array $queue = [
        //
    ];

    /**
     * The memorized results of the last parallel computation.
     *
     * @var mixed[]
     */
    protected array $results = [
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

        $workers = $workers ?? CPU::cores();

        $this->pool = new ContextWorkerPool($workers);
    }

    /**
     * Return the number of background worker processes.
     *
     * @return int
     */
    public function workers() : int
    {
        return $this->pool->getLimit();
    }

    /**
     * Queue up a deferred task for backend processing.
     *
     * @param Task $task
     * @param callable(mixed,mixed):void $after
     * @param mixed $context
     * @internal
     */
    public function enqueue(Task $task, ?callable $after = null, $context = null) : void
    {
        $task = new CallableTask($task);

        $coroutine = $this->coroutine($task, $after, $context);

        $this->queue[] = $coroutine;
    }

    /**
     * The future for a particular task and callback.
     *
     * @template TResult
     * @template TReceive
     * @template TSend
     * @param AmpTask<TResult, TReceive, TSend> $task
     * @param callable(mixed,mixed):void $after
     * @param mixed $context
     * @return Future<mixed>
     * @internal
     */
    public function coroutine(AmpTask $task, ?callable $after = null, $context = null) : Future
    {
        return async(function () use ($context, $after, $task) {
            $result = $this->pool->submit($task)->await();

            if ($after) {
                $after($result, $context);
            }

            return $result;
        });
    }

    /**
     * Process the queue and return the results.
     *
     * @return mixed[]
     * @internal
     */
    public function process() : array
    {
        $this->results = Future\await($this->queue);

        $this->queue = [];

        return $this->results;
    }

    /**
     * Flush the queue and clear the memorized results.
     *
     * @internal
     */
    public function flush() : void
    {
        $this->queue = $this->results = [];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     * @internal
     */
    public function __toString() : string
    {
        return "Amp (workers: {$this->workers()})";
    }
}
