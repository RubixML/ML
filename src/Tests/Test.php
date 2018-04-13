<?php

namespace Rubix\Engine\Tests;

use Rubix\Engine\Tests\Loggers\Screen;
use Rubix\Engine\Tests\Loggers\Logger;

abstract class Test
{
    const EPSILON = 1e-8;

    /**
     * The logging interface to use.
     *
     * @var \Rubix\Engine\Tests\Loggers\Logger
     */
    protected $logger;

    /**
     * @param  \Rubix\Engine\Tests\Loggers\Logger  $logger
     * @return void
     */
    public function __construct(Logger $logger = null)
    {
        if (!isset($logger)) {
            $logger = new Screen();
        }

        $this->logger = $logger;
    }

    /**
     * Score the test.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @return float
     */
    abstract public function score(array $predictions, array $outcomes) : float;
}
