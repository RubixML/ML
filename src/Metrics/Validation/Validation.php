<?php

namespace Rubix\Engine\Metrics\Validation;

interface Validation
{
    const EPSILON = 1e-8;

    /**
     * Score a group of predictions and return the value.
     *
     * @param  array  $predictions
     * @param  array  $labels
     * @return float
     */
    public function score(array $predictions, array $labels) : float;
}
