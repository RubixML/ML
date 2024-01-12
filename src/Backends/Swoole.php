<?php

namespace Rubix\ML\Backends;

use Rubix\ML\Backends\Tasks\Task;
use Rubix\ML\Serializers\Serializer;
use Rubix\ML\Serializers\Igbinary;
use Rubix\ML\Serializers\Native;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Swoole\Atomic;
use Swoole\Process;

use function Swoole\Coroutine\run;

/**
 * Swoole
 *
 * Works both with Swoole and OpenSwoole.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 */
class Swoole implements Backend
{
    /**
     * Swoole accepts values between 0.2 and 1
     */
    const CONFLICT_PROPORTION = 0.25;

    /**
     * The queue of tasks to be processed in parallel.
     */
    protected array $queue = [];

    private int $cpus;

    private Serializer $serializer;

    public function __construct(?Serializer $serializer = null)
    {
        $this->cpus = swoole_cpu_num();

        if ($serializer) {
            $this->serializer = $serializer;
        } else {
            if (ExtensionIsLoaded::with('igbinary')->passes()) {
                $this->serializer = new Igbinary();
            } else {
                $this->serializer = new Native();
            }
        }
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
        $results = [];

        $finishedTasksAtomic = new Atomic();
        $waitingAtomic = new Atomic();

        $scheduledTasksTotal = count($this->queue);
        $workerProcesses = [];

        $currentCpu = 0;

        while (($queueItem = array_shift($this->queue))) {
            $workerProcess = new Process(
                callback: function (Process $worker) use ($finishedTasksAtomic, $queueItem, $scheduledTasksTotal, $waitingAtomic) {
                    try {
                        $worker->exportSocket()->send(igbinary_serialize($queueItem()));
                    } finally {
                        $finishedTasksAtomic->add(1);

                        if ($scheduledTasksTotal <= $finishedTasksAtomic->get()) {
                            $waitingAtomic->wakeup();
                        }
                    }
                },
                enable_coroutine: true,
                pipe_type: 1,
                redirect_stdin_and_stdout: false,
            );

            $workerProcess->setAffinity([
                $currentCpu,
            ]);
            $workerProcess->setBlocking(false);
            $workerProcess->start();

            $workerProcesses[] = $workerProcess;

            $currentCpu = ($currentCpu + 1) % $this->cpus;
        }

        $waitingAtomic->wait(-1);

        run(function () use (&$results, $workerProcesses) {
            foreach ($workerProcesses as $workerProcess) {
                $receivedData = $workerProcess->exportSocket()->recv();
                $unserialized = igbinary_unserialize($receivedData);

                $results[] = $unserialized;
            }
        });

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
