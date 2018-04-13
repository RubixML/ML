<?php

namespace Rubix\Engine\Tests\Loggers;

class Stack implements Logger
{
    /**
     * The stack of loggers to run for each event.
     *
     * @var resource
     */
    protected $loggers = [
        //
    ];

    /**
     * @param  array  $loggers
     * @return void
     */
    public function __construct(array $loggers)
    {
        foreach ($loggers as $logger) {
            $this->addLogger($logger);
        }
    }

    /**
     * Run the log stack.
     *
     * @param  string  $message
     * @return void
     */
    public function log(string $message) : void
    {
        foreach ($this->loggers as $logger) {
            $logger->log($message);
        }
    }

    /**
     * Add a logger to the logging stack.
     *
     * @param  \Rubix\Engine\Loggers\Logger  $logger
     * @return void
     */
    protected function addLogger(Logger $logger) : void
    {
        $this->loggers[] = $logger;
    }
}
