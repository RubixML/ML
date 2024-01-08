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
     * Swoole accepts values between 0.2 and 1
     */
    const HASH_COLLISIONS_ALLOWED = 0.25;

    /**
     * The queue of tasks to be processed in parallel.
     */
    protected array $queue = [];

    private int $cpus;

    private int $serialiedRowLength;

    private Serializer $serializer;

    public function __construct(
        int $serialiedRowLength = 65536,
        ?Serializer $serializer = null,
    ) {
        if ($serializer) {
            $this->serializer = $serializer;
        } else {
            if (extension_loaded('igbinary')) {
                $this->serializer = new IgbinarySerializer();
            } else {
                $this->serializer = new PhpSerializer();
            }
        }

        $this->serialiedRowLength = $serialiedRowLength;
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
        $resultsTable = new Table(count($this->queue), self::HASH_COLLISIONS_ALLOWED);
        $resultsTable->column('result', Table::TYPE_STRING, $this->serialiedRowLength);
        $resultsTable->create();

        $workersTable = new Table($this->cpus, self::HASH_COLLISIONS_ALLOWED);
        $workersTable->column('working', Table::TYPE_INT);
        $workersTable->create();

        $pool = new Pool($this->cpus);

        $pool->on('WorkerStart', function (Pool $pool, $workerId) use ($resultsTable, $workersTable) {
            try {
                $process = $pool->getProcess();

                // Worker id is an integer that goes from 0 to the number of
                // workers. There are "$this->cpu" workers spawned, so worker
                // id should correspond to a specific core.
                $process->setAffinity([$workerId]);

                if (!$workersTable->exist($workerId)) {
                    if (!$workersTable->set($workerId, [
                        'working' => 1,
                    ])) {
                        throw new RuntimeException('Unable to store worker status in the shared memory table');
                    }

                    for ($i = $workerId; $i < count($this->queue); $i += $this->cpus) {
                        if (!$resultsTable->exist($i)) {
                            $result = $this->queue[$i]();
                            $serialized = $this->serializer->serialize($result);

                            if (!$resultsTable->set($i, [
                                'result' => $serialized,
                            ])) {
                                throw new RuntimeException('Unable to store task result in the shared memory table');
                            }
                        }
                    }
                }
            } finally {
                // Shuts down only the current worker. Tells Pool to not
                // create a new worker
                $pool->shutdown();
            }
        });

        // This is blocking, waits until all processes finish.
        $pool->start();

        $results = [];

        for ($i = 0; $i < count($this->queue); $i += 1) {
            $serialized = $resultsTable->get($i, 'result');
            $unserialized = $this->serializer->unserialize($serialized);

            if (false === $unserialized) {
                // Task needs to be repeated due to hash collision in the Table
                // That should be at most HASH_COLLISIONS_ALLOWED, usually less
                //
                // If 'false' was serialized, then the task will be redone
                // unnecessarily. That is the price we have to pay for the lack
                // of proper error handling in 'unserialize'. If you disagree
                // or have some better idea, please open an issue on GitHub. ;)
                $results[] = $this->queue[$i]();
            } else {
                $results[] = $unserialized;
            }
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
}
