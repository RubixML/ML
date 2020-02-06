<?php

namespace Rubix\ML\CrossValidation\Metrics;

interface Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array;

    /**
     * The estimator types that this metric is compatible with.
     *
     * @return \Rubix\ML\EstimatorType[]
     */
    public function compatibility() : array;

    /**
     * Score a set of predictions.
     *
     * @param (string|int|float)[] $predictions
     * @param (string|int|float)[] $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float;
}
