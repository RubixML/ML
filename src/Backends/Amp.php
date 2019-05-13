<?php

namespace Rubix\ML\Backends;

use Amp\Loop;
use Rubix\ML\Deferred;
use Rubix\ML\Other\Helpers\CPU;
use Amp\Parallel\Worker\DefaultPool;
use Amp\Parallel\Worker\CallableTask;
use InvalidArgumentException;
use Closure;

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
     * Automatically build an Amp backend based on processor core count.
     *
     * @return self
     */
    public static function autotune() : self
    {
        return new self(CPU::cores());
    }

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
     * Queue up a deferred computation for backend processing.
     *
     * @param \Rubix\ML\Deferred $deferred
     * @param \Closure|null $after
     * @throws \InvalidArgumentException
     */
    public function enqueue(Deferred $deferred, ?Closure $after = null) : void
    {
        $task = new CallableTask([$deferred, 'compute'], []);

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
     * Process the queue and return the results.
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

    /**
     * Flush the queue.
     */
    public function flush() : void
    {
        $this->queue = [];
    }
}
