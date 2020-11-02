<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Stringable;

interface Metric extends Stringable
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return array{float,float}
     */
    public function range() : array;

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
