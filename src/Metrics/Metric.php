<?php

namespace Rubix\Engine\Metrics;

interface Metric
{
    const EPSILON = 1e-8;

    /**
     * Score a group of predictions and return the value.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @return float
     */
    public function score(array $predictions, array $outcomes) : float;

    /**
     * Should this metric be minimized?
     *
     * @return bool
     */
    public function minimize() : bool;
}
