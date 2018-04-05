<?php

namespace Rubix\Engine\Tests;

use Rubix\Engine\Estimator;
use Rubix\Engine\SupervisedDataset;

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
     * @param  \Rubix\Engine\SupervisedDataset  $data
     * @return bool
     */
    abstract public function test(SupervisedDataset $data) : bool;
}
