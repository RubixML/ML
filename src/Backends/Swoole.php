<?php

namespace Rubix\ML\Backends;

use Rubix\ML\Backends\Tasks\Task;
use Rubix\ML\Helpers\CPU;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Swoole\Atomic;
use Swoole\Coroutine\System;
use Swoole\Process;

use function Swoole\Coroutine\run;
use function call_user_func;
use function method_exists;
use function array_fill;
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
        $maxMessageLength = new Atomic(0);

        $results = array_fill(0, count($this->queue), null);

        $prepared = [];

        foreach ($this->queue as $item) {
            $prepared[] = new Process(
                function (Process $worker) use ($maxMessageLength, $item) {
                    $serialized = call_user_func($this->serialize, $item());

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
        }

        $currentCpu = $next = 0;
        $running = [];

        $start = function (int $index) use (&$running, &$currentCpu, &$next, $prepared) {
            $process = $prepared[$index];

            if (method_exists(Process::class, 'setAffinity')) {
                Process::setAffinity([$currentCpu]);
            }

            $process->setBlocking(false);
            $process->start();

            $running[$process->pid] = [$index, $process];

            ++$currentCpu;

            $currentCpu %= $this->workers;

            ++$next;
        };

        while (count($running) < $this->workers and $next < count($prepared)) {
            $start($next);
        }

        while ($running) {
            run(function () use (&$running, &$results, $maxMessageLength) {
                $status = System::wait();

                $index = $running[$status['pid']][0];
                $process = $running[$status['pid']][1];

                unset($running[$status['pid']]);

                if (0 !== $status['code']) {
                    throw new RuntimeException('Worker process exited with an error.');
                }

                $socket = $process->exportSocket();

                if ($socket->isClosed()) {
                    throw new RuntimeException('Coroutine socket is already closed.');
                }

                $receivedData = $socket->recv($maxMessageLength->get());

                $results[$index] = call_user_func($this->unserialize, $receivedData);
            });

            if ($next < count($prepared)) {
                $start($next);
            }
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
