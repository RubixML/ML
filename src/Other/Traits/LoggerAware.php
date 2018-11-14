<?php

namespace Rubix\ML\Other\Traits;

use Rubix\ML\Verbose;
use Rubix\ML\MetaEstimator;
use Psr\Log\LoggerInterface;

/**
 * Logger Aware
 *
 * This trait fulfills the requirements of the Verbose interface and
 * is suitable for most estimators including meta-estimators that
 * pass the logger instance to the base estimator.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
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

    /**
     * Return if the logger is logging or not.
     * 
     * @var bool
     */
    public function logging() : bool
    {
        return isset($this->logger);
    }
}