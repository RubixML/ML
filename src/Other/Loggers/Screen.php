<?php

namespace Rubix\ML\Other\Loggers;

use Psr\Log\LoggerInterface;
use Psr\Log\LogLevel;

/**
 * Screen
 *
 * A logger that outputs to the php standard output.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Screen implements LoggerInterface
{
    /**
     * The channel name that appears on each line.
     * 
     * @var string
     */
    protected $channel;

    /**
     * Should we show timestamps?
     * 
     * @var bool
     */
    protected $timestamps;

    /**
     * @param  string  $channel
     * @param  bool  $timestamps
     * @return void
     */
    public function __construct(string $channel = 'default', bool $timestamps = true)
    {
        $this->channel = trim($channel);
        $this->timestamps = $timestamps;
    }

    /**
     * System is unusable.
     *
     * @param  string  $message
     * @param  array  $context
     * @return void
     */
    public function emergency($message, array $context = []) : void
    {
        $this->log(LogLevel::EMERGENCY, $message, $context);
    }

    /**
     * Action must be taken immediately.
     *
     * Example: Entire website down, database unavailable, etc. This should
     * trigger the SMS alerts and wake you up.
     *
     * @param  string  $message
     * @param  array  $context
     * @return void
     */
    public function alert($message, array $context = []) : void
    {
        $this->log(LogLevel::ALERT, $message, $context);
    }

    /**
     * Critical conditions.
     *
     * Example: Application component unavailable, unexpected exception.
     *
     * @param  string $message
     * @param  array $context
     * @return void
     */
    public function critical($message, array $context = []) : void
    {
        $this->log(LogLevel::CRITICAL, $message, $context);
    }

    /**
     * Runtime errors that do not require immediate action but should typically
     * be logged and monitored.
     *
     * @param  string  $message
     * @param  array  $context
     * @return void
     */
    public function error($message, array $context = []) : void
    {
        $this->log(LogLevel::ERROR, $message, $context);
    }

    /**
     * Exceptional occurrences that are not errors.
     *
     * Example: Use of deprecated APIs, poor use of an API, undesirable things
     * that are not necessarily wrong.
     *
     * @param  string  $message
     * @param  array  $context
     * @return void
     */
    public function warning($message, array $context = []) : void
    {
        $this->log(LogLevel::WARNING, $message, $context);
    }

    /**
     * Normal but significant events.
     *
     * @param  string  $message
     * @param  array  $context
     * @return void
     */
    public function notice($message, array $context = []) : void
    {
        $this->log(LogLevel::NOTICE, $message, $context);
    }

    /**
     * Interesting events.
     *
     * Example: User logs in, SQL logs.
     *
     * @param  string  $message
     * @param  array  $context
     * @return void
     */
    public function info($message, array $context = []) : void
    {
        $this->log(LogLevel::INFO, $message, $context);
    }

    /**
     * Detailed debug information.
     *
     * @param  string  $message
     * @param  array  $context
     * @return void
     */
    public function debug($message, array $context = []) : void
    {
        $this->log(LogLevel::DEBUG, $message, $context);
    }

    /**
     * Logs with an arbitrary level.
     *
     * @param  mixed  $level
     * @param  string  $message
     * @param  array  $context
     * @return void
     */
    public function log($level, $message, array $context = []) : void
    {
        $prefix = '';

        if ($this->timestamps) {
            $prefix .= '[' . date('Y-m-d H:i:s') . '] ';
        }

        $prefix .=  $this->channel . '.' . strtoupper((string) $level) . ': ';

        echo $prefix . $message . PHP_EOL;
    }
}