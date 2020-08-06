<?php

namespace Rubix\ML\Other\Loggers;

/**
 * Screen
 *
 * A logger that displays log messages to the standard output.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Screen extends Logger
{
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
    public function __construct(string $channel = '', string $format = 'Y-m-d H:i:s')
    {
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
        $prefix = '';

        if ($this->format) {
            $prefix .= '[' . date($this->format) . '] ';
        }

        if ($this->channel) {
            $prefix .= $this->channel . '.';
        }

        $prefix .= strtoupper((string) $level);

        echo $prefix . ': ' . trim($message) . PHP_EOL;
    }
}
