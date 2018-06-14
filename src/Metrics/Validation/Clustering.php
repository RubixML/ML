<?php

namespace Rubix\Engine\Metrics\Validation;

use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Clusterers\Clusterer;

interface Clustering extends Validation
{
    /**
     * Score a clusterer using a labeled testing set.
     *
     * @param  \Rubix\Engine\Clusterers\Clusterer  $estimator
     * @param  \Runix\Engine\Datasets\Labeled  $testing
     * @return float
     */
    public function score(Clusterer $estimator, Labeled $testing) : float;
}
