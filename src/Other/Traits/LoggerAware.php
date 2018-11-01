<?php

namespace Rubix\ML\Other\Traits;

use Psr\Log\LoggerInterface;

trait LoggerAware
{
    /**
     * The PSR-3 logger instance.
     * 
     * @var \Psr\Log\LoggerInterface
     */
    protected $logger;

    /**
     * Sets a logger instance on the object.
     *
     * @param  \Psr\Log\LoggerInterface  $logger
     * @return void
     */
    public function setLogger(LoggerInterface $logger) : void
    {
        $this->logger = $logger;
    }
}