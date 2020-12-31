<?php

namespace Rubix\ML\Other\Traits;

use Psr\Log\LoggerInterface;

/**
 * Logger Aware
 *
 * This trait fulfills the requirements of the Verbose interface and is suitable for most estimators.
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
     * Sets a PSR-3 logger instance.
     *
     * @param \Psr\Log\LoggerInterface $logger
     */
    public function setLogger(LoggerInterface $logger) : void
    {
        $this->logger = $logger;
    }

    /**
     * Return the PSR-3 logger instance.
     *
     * @return \Psr\Log\LoggerInterface|null
     */
    public function logger() : ?LoggerInterface
    {
        return $this->logger;
    }
}
