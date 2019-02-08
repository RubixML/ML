<?php

namespace Rubix\ML\CrossValidation\Metrics;

interface Metric
{
    const EPSILON = 1e-8;

    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array;

    /**
     * The estimator types that this metric is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array;

    /**
     * Score a set of predictions.
     *
     * @param array $predictions
     * @param array $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float;
}
