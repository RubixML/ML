<?php

namespace Rubix\ML\Traits;

use Psr\Log\LoggerInterface;

/**
 * Logger Aware
 *
 * This trait fulfills the requirements of the Verbose interface and is suitable for most implementations.
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
     * @var LoggerInterface|null
     */
    protected ?LoggerInterface $logger = null;

    /**
     * Sets a PSR-3 logger instance.
     *
     * @param LoggerInterface|null $logger
     */
    public function setLogger(?LoggerInterface $logger) : void
    {
        $this->logger = $logger;
    }

    /**
     * Return the PSR-3 logger instance.
     *
     * @return LoggerInterface|null
     */
    public function logger() : ?LoggerInterface
    {
        return $this->logger;
    }
}
