<?php

namespace Rubix\ML\Other\Loggers;

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
    public const DEFAULT_CHANNEL = 'main';
    
    public const DEFAULT_TIMESTAMP_FORMAT = 'Y-m-d H:i:s';
    
    /**
     * The channel name that appears on each line.
     *
     * @var string
     */
    protected $channel;

    /**
     * The format of the timestamp.
     *
     * @var string
     */
    protected $format;

    /**
     * @param string $channel
     * @param string $format
     */
    public function __construct(
        string $channel = self::DEFAULT_CHANNEL,
        string $format = self::DEFAULT_TIMESTAMP_FORMAT
    ) {
        $this->channel = trim($channel);
        $this->format = $format;
    }

    /**
     * Logs with an arbitrary level.
     *
     * @param mixed $level
     * @param string $message
     * @param mixed[] $context
     */
    public function log($level, $message, array $context = []) : void
    {
        $prefix = $this->format ? '[' . date($this->format) . '] ' : '';

        $prefix .= $this->channel . '.' . strtoupper((string) $level) . ': ';

        echo $prefix . trim($message) . PHP_EOL;
    }
}
