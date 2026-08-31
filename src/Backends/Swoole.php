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
use function call_user_func;
use function method_exists;
use function array_chunk;
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
     *
     * @var array<callable():mixed>
     */
    protected array $queue = [];

    /**
     * The number of workers available to the backend.
     *
     * @var int
     */
    protected int $workers;

    /**
     * The serialization function to use.
     *
     * @var callable
     */
    protected string $serialize;

    /**
     * The unserialization function to use.
     *
     * @var callable
     */
    protected string $unserialize;

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

        $hasIgbinary = ExtensionIsLoaded::with('igbinary')->passes();

        $this->workers = $workers ?? CPU::cores();
        $this->serialize = $hasIgbinary ? 'igbinary_serialize' : 'serialize';
        $this->unserialize = $hasIgbinary ? 'igbinary_unserialize' : 'unserialize';
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
        $results = [];
        $currentCpu = 0;

        $chunks = array_chunk($this->queue, $this->workers);

        foreach ($chunks as $batch) {
            $maxMessageLength = new Atomic(0);

            $processes = [];

            foreach ($batch as $queueItem) {
                $process = new Process(
                    function (Process $worker) use ($maxMessageLength, $queueItem) {
                        $serialized = call_user_func($this->serialize, $queueItem());

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

                if (method_exists(Process::class, 'setAffinity')) {
                    Process::setAffinity([$currentCpu]);
                }

                $process->setBlocking(false);
                $process->start();

                $processes[] = $process;

                ++$currentCpu;

                $currentCpu %= $this->workers;
            }

            run(function () use ($maxMessageLength, &$results, $processes) {
                foreach ($processes as $process) {
                    $status = $process->wait();

                    if (0 !== $status['code']) {
                        throw new RuntimeException('Worker process exited with an error.');
                    }

                    $socket = $process->exportSocket();

                    if ($socket->isClosed()) {
                        throw new RuntimeException('Coroutine socket is already closed.');
                    }

                    $maxMessageLengthValue = $maxMessageLength->get();

                    $receivedData = $socket->recv($maxMessageLengthValue);

                    $unserialized = call_user_func($this->unserialize, $receivedData);

                    $results[] = $unserialized;
                }
            });
        }

        $this->flush();

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
     * @return string
     */
    public function __toString() : string
    {
        return "Swoole (workers: {$this->workers})";
    }
}
