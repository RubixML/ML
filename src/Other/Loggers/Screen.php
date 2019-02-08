<?php

namespace Rubix\ML\Other\Loggers;

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
class Screen extends Logger
{
    const TIMESTAMP_FORMAT = 'Y-m-d H:i:s';
    
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
            $prefix .= '[' . date(self::TIMESTAMP_FORMAT) . '] ';
        }

        $prefix .=  $this->channel . '.' . strtoupper((string) $level) . ': ';

        echo $prefix . $message . PHP_EOL;
    }
}
