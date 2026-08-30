<?php

namespace Rubix\ML\Backends;

use Rubix\ML\Backends\Tasks\Task;
use Rubix\ML\Helpers\CPU;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Swoole\Atomic;
use Swoole\Process;

use function Swoole\Coroutine\run;
use function method_exists;
use function serialize;
use function unserialize;
use function strlen;

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

    /**
     * The number of workers available to the backend.
     *
     * @var int
     */
    protected int $workers;

    /**
     * Whether the igbinary extension is loaded.
     *
     * @var bool
     */
    protected bool $hasIgbinary;

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

        ExtensionIsLoaded::with('swoole')->check();

        $workers ??= CPU::cores();

        $hasIgbinary = ExtensionIsLoaded::with('igbinary')->passes();

        $this->workers = $workers;
        $this->hasIgbinary = $hasIgbinary;
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
        $this->queue[] = function () use ($task, $after, $context) {
            $result = $task();

            if ($after) {
                $after($result, $context);
            }

            return $result;
        };
    }

    /**
     * Return the number of concurrent worker processes.
     *
     * @internal
     *
     * @return int
     */
    public function workers() : int
    {
        return $this->workers;
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
        $maxMessageLength = new Atomic(0);
        $workerProcesses = [];
        $currentCpu = 0;

        foreach ($this->queue as $index => $queueItem) {
            $workerProcess = new Process(
                function (Process $worker) use ($maxMessageLength, $queueItem) {
                    $serialized = $this->serialize($queueItem());

                    $length = strlen($serialized);

                    $currentMaxLength = $maxMessageLength->get();

                    if ($length > $currentMaxLength) {
                        $maxMessageLength->set($length);
                    }

                    $worker->exportSocket()->send($serialized);
                },
                false, // Redirect_stdin_and_stdout
                SOCK_DGRAM, // Pipe type
                true, // Enable coroutine
            );

            if (method_exists($workerProcess, 'setAffinity')) {
                $workerProcess->setAffinity([$currentCpu]);
            }

            $workerProcess->setBlocking(false);
            $workerProcess->start();

            $workerProcesses[$index] = $workerProcess;

            $currentCpu = ($currentCpu + 1) % $this->workers;
        }

        $results = [];

        run(function () use ($maxMessageLength, &$results, $workerProcesses) {
            foreach ($workerProcesses as $workerProcess) {
                $status = $workerProcess->wait();

                if (0 !== $status['code']) {
                    throw new RuntimeException('Worker process exited with an error.');
                }

                $socket = $workerProcess->exportSocket();

                if ($socket->isClosed()) {
                    throw new RuntimeException('Coroutine socket is closed.');
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

    /**
     * Shut down the backend.
     *
     * @internal
     */
    public function shutdown() : void
    {
        // No-op for the Swoole backend.
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @param mixed $data
     * @return string
     */
    protected function serialize(mixed $data) : string
    {
        return $this->hasIgbinary
            ? igbinary_serialize($data)
            : serialize($data);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @param string $serialized
     * @return mixed
     */
    protected function unserialize(string $serialized) : mixed
    {
        return $this->hasIgbinary
            ? igbinary_unserialize($serialized)
            : unserialize($serialized);
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
        return "Swoole (workers: {$this->workers})";
    }
}
