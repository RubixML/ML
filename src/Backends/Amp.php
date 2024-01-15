<?php

namespace Rubix\ML\Backends;

use Amp\Loop;
use Rubix\ML\Helpers\CPU;
use Rubix\ML\Backends\Tasks\Task;
use Amp\Parallel\Worker\DefaultPool;
use Amp\Parallel\Worker\CallableTask;
use Amp\Parallel\Worker\Task as AmpTask;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Generator;

use function Amp\call;
use function Amp\Promise\all;

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
class Amp implements Backend
{
    /**
     * The worker pool.
     *
     * @var \Amp\Parallel\Worker\Pool
     */
    protected \Amp\Parallel\Worker\Pool $pool;

    /**
     * The queue of coroutines to be processed in parallel.
     *
     * @var \Amp\Promise<mixed>[]
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

        $this->pool = new DefaultPool($workers);
    }

    /**
     * Return the number of background worker processes.
     *
     * @return int
     */
    public function workers() : int
    {
        return $this->pool->getMaxSize();
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
    public function enqueue(Task $task, ?callable $after = null, $context = null) : void
    {
        $task = new CallableTask($task, []);

        $coroutine = call([$this, 'coroutine'], $task, $after, $context);

        $this->queue[] = $coroutine;
    }

    /**
     * The coroutine for a particular task and callback.
     *
     * @internal
     *
     * @param AmpTask $task
     * @param callable(mixed,mixed):void $after
     * @param mixed $context
     * @return \Generator<\Amp\Promise>
     */
    public function coroutine(AmpTask $task, ?callable $after = null, $context = null) : Generator
    {
        $result = yield $this->pool->enqueue($task);

        if ($after) {
            $after($result, $context);
        }

        return $result;
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
        Loop::run([$this, 'gather']);

        $this->queue = [];

        return $this->results;
    }

    /**
     * Gather and memorize the results from the worker pool.
     *
     * @internal
     *
     * @return \Generator<\Amp\Promise>
     */
    public function gather() : Generator
    {
        $this->results = yield all($this->queue);
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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Amp (workers: {$this->workers()})";
    }
}
