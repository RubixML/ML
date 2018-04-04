<?php

namespace Rubix\Engine\Tests;

use Rubix\Engine\Estimator;

abstract class Test
{
    /**
     * The estimator being tested.
     *
     * @var \Rubix\Engine\Estimator
     */
    protected $estimator;

    /**
     * Prepare an estimator instance for testing.
     *
     * @param  \Rubix\Engine\Estimator  $estimator
     * @return self
     */
    public function load(Estimator $estimator) : self
    {
        $this->estimator = $estimator;

        return $this;
    }

    /**
     * Run the test.
     *
     * @return mixed
     */
    abstract public function test(array $samples, array $outcomes);
}
