<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;

interface Validation
{
    const EPSILON = 1e-8;

    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return array
     */
    public function range() : array;

    /**
     * Score an estimator using a labeled testing set.
     *
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Dataset  $testing
     * @return float
     */
    public function score(Estimator $estimator, Dataset $testing) : float;
}
