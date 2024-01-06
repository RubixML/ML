<?php

namespace Rubix\ML\Backends\Swoole;

use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Tasks\Task;
use RuntimeException;
use Swoole\Process\Pool;
use Swoole\Table;

/**
 * Swoole
 *
 * Works both with Swoole and OpenSwoole.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 */
class Process implements Backend
{
    /**
     * The queue of tasks to be processed in parallel.
     */
    protected array $queue = [];

    private $cpus;

    public function __construct()
    {
        $this->cpus = swoole_cpu_num();
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
        $resultsTable = new Table(count($this->queue) * 2);
        $resultsTable->column('result', Table::TYPE_FLOAT);
        $resultsTable->create();

        $workersTable = new Table($this->cpus);
        $workersTable->column('working', Table::TYPE_INT);
        $workersTable->create();

        $pool = new Pool($this->cpus);
        $pool->on('WorkerStart', function (Pool $pool, $workerId) use ($resultsTable, $workersTable) {
            try {
                if (!$workersTable->exist($workerId)) {
                    $workersTable->set($workerId, [
                        'working' => 1,
                    ]);

                    for ($i = $workerId; $i < count($this->queue); $i += $this->cpus) {
                        if (!$resultsTable->exist($i)) {
                            $resultsTable->set($i, [
                                'result' => $this->queue[$i](),
                            ]);
                        }
                    }
                }
            } finally {
                $pool->shutdown();
            }
        });
        $pool->start();

        $results = [];

        for ($i = 0; $i < count($this->queue); $i += 1) {
            $results[] = $resultsTable->get($i, 'result');
        }

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
        return 'Swoole\\Process';
    }

    private static function nextPowerOf2($number)
    {
        return pow(2, ceil(log($number,2)));
    }
}
