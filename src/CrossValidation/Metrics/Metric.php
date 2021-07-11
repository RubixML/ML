<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Stringable;
use Tuple;

interface Metric extends Stringable
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return \Tuple{float,float}
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
     * @param list<string|int|float> $predictions
     * @param list<string|int|float> $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float;
}
