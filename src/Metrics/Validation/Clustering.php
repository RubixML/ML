<?php

namespace Rubix\ML\Metrics\Validation;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Clusterers\Clusterer;

interface Clustering extends Validation
{
    /**
     * Score a clusterer using a labeled testing set.
     *
     * @param  \Rubix\ML\Clusterers\Clusterer  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return float
     */
    public function score(Clusterer $estimator, Labeled $testing) : float;
}
