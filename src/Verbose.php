<?php

namespace Rubix\ML;

use Psr\Log\LoggerAwareInterface;

interface Verbose extends LoggerAwareInterface
{
    /**
     * Return if the logger is logging or not.
     * 
     * @var bool
     */
    public function logging() : bool;
}