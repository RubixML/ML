<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Stringable;

interface Metric extends Stringable
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return \Rubix\ML\Tuple{float,float}
     */
    public function range() : Tuple;

    /**
     * The estimator types that this metric is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\EstimatorType>
     */
    public function compatibility() : array;

    /**
     * Score a set of predictions.
     *
     * @param ScoreInput $input
     * @return float
     */
    public function score(ScoreInput $input) : float;
}
