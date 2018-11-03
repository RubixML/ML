<?php

namespace Rubix\ML\Other\Traits;

use Rubix\ML\Verbose;
use Rubix\ML\MetaEstimator;
use Psr\Log\LoggerInterface;

trait LoggerAware
{
    /**
     * The PSR-3 logger instance.
     * 
     * @var \Psr\Log\LoggerInterface|null
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
        if ($this instanceof MetaEstimator) {
            $estimator = $this->estimator();
            
            if ($estimator instanceof Verbose) {
                $estimator->setLogger($logger);
            }
        }

        $this->logger = $logger;
    }
}