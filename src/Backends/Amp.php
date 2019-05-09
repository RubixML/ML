<?php

namespace Rubix\ML\Backends;

use Amp\Loop;
use Amp\Parallel\Worker\DefaultPool;
use Amp\Parallel\Worker\CallableTask;
use InvalidArgumentException;

use function Amp\call;
use function Amp\Promise\all;

/**
 * Amp
 *
 * Amp Parallel is a multiprocessing subsystem that requires no extensions.
 * It uses a non-blocking concurrency framework that implements coroutines
 * using PHP generator functions under the hood.
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
    protected $pool;

    /**
     * The queue of coroutines.
     *
     * @var array
     */
    protected $queue = [
        //
    ];

    /**
     * @param int $workers
     * @throws \InvalidArgumentException
     */
    public function __construct(int $workers = 4)
    {
        if ($workers < 1) {
            throw new InvalidArgumentException('Number of workers'
                . " must be greater than 0, $workers given.");
        }

        $this->pool = new DefaultPool($workers);
    }

    /**
     * Queue up a function for backend processing.
     *
     * @param callable $function
     * @param array $args
     * @param callable $after
     */
    public function enqueue(callable $function, array $args = [], ?callable $after = null) : void
    {
        $task = new CallableTask($function, $args);

        $coroutine = call(function () use ($task, $after) {
            $result = yield $this->pool->enqueue($task);

            if ($after) {
                $after($result);
            }

            return $result;
        });

        $this->queue[] = $coroutine;
    }

    /**
     * Process the queue.
     *
     * @return array
     */
    public function process() : array
    {
        $results = [];

        Loop::run(function () use (&$results) {
            $results = yield all($this->queue);
        });

        $this->queue = [];

        return $results;
    }
}
