<?php

namespace Rubix\ML\Backends;

use Rubix\ML\Backends\Tasks\Task;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\SwooleExtensionIsLoaded;
use RuntimeException;
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
     * The queue of tasks to be processed in parallel.
     */
    protected array $queue = [];

    private int $cpus;

    private int $hasIgbinary;

    public function __construct()
    {
        SwooleExtensionIsLoaded::create()->check();

        $this->cpus = swoole_cpu_num();
        $this->hasIgbinary = ExtensionIsLoaded::with('igbinary')->passes();
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

        $maxMessageLength = new Atomic(0);
        $workerProcesses = [];

        $currentCpu = 0;

        foreach ($this->queue as $index => $queueItem) {
            $workerProcess = new Process(
                function (Process $worker) use ($maxMessageLength, $queueItem) {
                    $serialized = $this->serialize($queueItem());

                    $serializedLength = strlen($serialized);
                    $currentMaxSerializedLength = $maxMessageLength->get();

                    if ($serializedLength > $currentMaxSerializedLength) {
                        $maxMessageLength->set($serializedLength);
                    }

                    $worker->exportSocket()->send($serialized);
                },
                // redirect_stdin_and_stdout
                false,
                // pipe_type
                SOCK_DGRAM,
                // enable_coroutine
                true,
            );

            $workerProcess->setAffinity([$currentCpu]);
            $workerProcess->setBlocking(false);
            $workerProcess->start();

            $workerProcesses[$index] = $workerProcess;

            $currentCpu = ($currentCpu + 1) % $this->cpus;
        }

        run(function () use ($maxMessageLength, &$results, $workerProcesses) {
            foreach ($workerProcesses as $index => $workerProcess) {
                $status = $workerProcess->wait();

                if (0 !== $status['code']) {
                    throw new RuntimeException('Worker process exited with an error');
                }

                $socket = $workerProcess->exportSocket();

                if ($socket->isClosed()) {
                    throw new RuntimeException('Coroutine socket is closed');
                }

                $maxMessageLengthValue = $maxMessageLength->get();

                $receivedData = $socket->recv($maxMessageLengthValue);
                $unserialized = $this->unserialize($receivedData);

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

    private function serialize(mixed $data) : string
    {
        if ($this->hasIgbinary) {
            return igbinary_serialize($data);
        }

        return serialize($data);
    }

    private function unserialize(string $serialized) : mixed
    {
        if ($this->hasIgbinary) {
            return igbinary_unserialize($serialized);
        }

        return unserialize($serialized);
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
        return 'Swoole';
    }
}
